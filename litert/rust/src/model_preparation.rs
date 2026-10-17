// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

use std::collections::HashSet;

// Field positions and enum values follow tflite/converter/schema/schema.fbs.
const DEQUANTIZE: u32 = 6;
const MAX_POOL_2D: u32 = 17;
const MAXIMUM: u32 = 55;
const STABLEHLO_REDUCE_WINDOW: u32 = 198;
const FLOAT32: u8 = 0;
const FLOAT16: u8 = 1;

pub(crate) fn prepare(data: &mut Vec<u8>) -> Option<()> {
    let plan = plan(data)?;
    data.extend_from_slice(&plan.appended);
    for (position, value) in plan.words {
        data[position..position + 4].copy_from_slice(&value.to_le_bytes());
    }
    for (position, value) in plan.bytes {
        data[position] = value;
    }
    Some(())
}

#[derive(Default)]
struct Plan {
    words: Vec<(usize, u32)>,
    bytes: Vec<(usize, u8)>,
    appended: Vec<u8>,
}

struct Graph {
    entries: usize,
    operators: Vec<usize>,
}

struct Pool {
    input: u32,
    output: u32,
    padding: u32,
    height: u32,
    width: u32,
    stride_h: u32,
    stride_w: u32,
}

fn plan(data: &[u8]) -> Option<Plan> {
    if data.get(4..8)? != b"TFL3" {
        return None;
    }
    let model = Table { data, position: word(data, 0)? as usize };
    if model.u32(0)? != 3 {
        return None;
    }
    let codes = model.vector(1, 4)?;
    let subgraphs = model.vector(2, 4)?;
    let buffers = model.vector(4, 4)?;
    let mut graphs = Vec::new();
    let mut pools = Vec::new();
    let mut unsupported_pool_codes = HashSet::new();
    for graph_index in 0..subgraphs.len() {
        let graph = subgraphs.table(graph_index)?;
        let operators = graph.vector(3, 4)?;
        let tensors = graph.vector(0, 4)?;
        let graph_inputs = graph.vector(1, 4)?.words()?;
        let mut produced = HashSet::new();
        for index in 0..operators.len() {
            produced.extend(operators.table(index)?.vector(2, 4)?.words()?);
        }
        let mut constant_ops = Vec::new();
        let mut remaining_ops = Vec::new();
        for index in 0..operators.len() {
            let op = operators.table(index)?;
            let code_index = op.u32(0)?;
            let code_table = codes.table(code_index as usize)?;
            let code = opcode(code_table)?;
            let inputs = op.vector(1, 4)?;
            let outputs = op.vector(2, 4)?;
            let constant_dequantize =
                if code == DEQUANTIZE && inputs.len() == 1 && outputs.len() == 1 {
                    let input_index = inputs.word(0)?;
                    let input = tensors.table(input_index as usize)?;
                    let output = tensors.table(outputs.word(0)? as usize)?;
                    input.u8(1)? == FLOAT16
                        && output.u8(1)? == FLOAT32
                        && input.u8(5)? == 0
                        && !graph_inputs.contains(&input_index)
                        && !produced.contains(&input_index)
                        && constant_data(input, buffers).is_some_and(|bytes| !bytes.is_empty())
                } else {
                    false
                };
            if constant_dequantize {
                constant_ops.push(op.position);
            } else {
                remaining_ops.push(op.position);
            }
            if code == STABLEHLO_REDUCE_WINDOW {
                let constant_initial = inputs.len() == 2
                    && !graph_inputs.contains(&inputs.word(1)?)
                    && !produced.contains(&inputs.word(1)?);
                let pool = constant_initial
                    .then(|| max_pool(op, tensors, buffers, subgraphs, codes))
                    .flatten();
                if let Some(pool) = pool {
                    pools.push((code_index, code_table.position, op.position, pool));
                } else {
                    unsupported_pool_codes.insert(code_table.position);
                }
            }
        }
        constant_ops.extend(remaining_ops);
        graphs.push(Graph { entries: operators.position, operators: constant_ops });
    }

    let mut plan = Plan::default();
    let mut replacements = Vec::new();
    let mut rewritten_codes = HashSet::new();
    for (code_index, code_position, position, pool) in pools {
        // An opcode table may be shared by several reductions. Reuse it only
        // when every use is the same supported max-pooling operation.
        if unsupported_pool_codes.contains(&code_position) {
            continue;
        }
        let replacement = append_pool(&mut plan.appended, data.len(), code_index, pool)?;
        replacements.push((position, replacement));
        if rewritten_codes.insert(code_position) {
            let code = codes.table(code_index as usize)?;
            plan.words.push((code.field(3)?, MAX_POOL_2D));
            if let Some(field) = code.field(0) {
                plan.bytes.push((field, MAX_POOL_2D as u8));
            }
            if let Some(field) = code.field(2) {
                plan.words.push((field, 1));
            }
        }
    }
    for graph in graphs {
        for (index, position) in graph.operators.into_iter().enumerate() {
            let target = replacements
                .iter()
                .find(|(old, _)| *old == position)
                .map_or(position, |(_, replacement)| *replacement);
            let entry = graph.entries.checked_add(index.checked_mul(4)?)?;
            let relative = u32::try_from(target.checked_sub(entry)?).ok()?;
            if word(data, entry)? != relative {
                plan.words.push((entry, relative));
            }
        }
    }
    for (position, _) in &plan.words {
        read::<4>(data, *position)?;
    }
    Some(plan)
}

fn max_pool(
    op: Table<'_>,
    tensors: Vector<'_>,
    buffers: Vector<'_>,
    subgraphs: Vector<'_>,
    codes: Vector<'_>,
) -> Option<Pool> {
    // BuiltinOptions2::StablehloReduceWindowOptions.
    if op.u8(11)? != 13 {
        return None;
    }
    let inputs = op.vector(1, 4)?.words()?;
    let outputs = op.vector(2, 4)?.words()?;
    if inputs.len() != 2 || outputs.len() != 1 {
        return None;
    }
    let input = tensors.table(inputs[0] as usize)?;
    let initial = tensors.table(inputs[1] as usize)?;
    let output = tensors.table(outputs[0] as usize)?;
    if input.u8(1)? != FLOAT32
        || output.u8(1)? != FLOAT32
        || initial.u8(1)? != FLOAT32
        || initial.u8(5)? != 0
        || constant_data(initial, buffers)? != f32::MIN.to_le_bytes()
        || !initial.vector(0, 4)?.words()?.is_empty()
    {
        return None;
    }
    let options = op.table(12)?;
    let window = options.vector(0, 8)?.longs()?;
    let strides = options.vector(1, 8)?.longs()?;
    let padding = options.vector(4, 8)?.longs()?;
    if window.len() != 4
        || strides.len() != 4
        || padding.len() != 8
        || window[0] != 1
        || window[3] != 1
        || strides[0] != 1
        || strides[3] != 1
        || options.vector(2, 8)?.longs()? != [1; 4]
        || options.vector(3, 8)?.longs()? != [1; 4]
        || padding[0..2] != [0, 0]
        || padding[6..8] != [0, 0]
    {
        return None;
    }
    let shape = input.vector(0, 4)?.words()?;
    let output_shape = output.vector(0, 4)?.words()?;
    if shape.len() != 4
        || output_shape.len() != 4
        || shape.iter().chain(&output_shape).any(|&d| d == 0 || d > i32::MAX as u32)
        || shape[0] != output_shape[0]
        || shape[3] != output_shape[3]
    {
        return None;
    }
    let mut same = true;
    let mut valid = true;
    for axis in 1..3 {
        let size = i64::from(shape[axis]);
        let stride = strides[axis];
        let filter = window[axis];
        if !(1..=i64::from(i32::MAX)).contains(&stride)
            || !(1..=i64::from(i32::MAX)).contains(&filter)
        {
            return None;
        }
        let same_size = (size + stride - 1) / stride;
        let total_padding = ((same_size - 1) * stride + filter - size).max(0);
        same &= padding[axis * 2] == total_padding / 2
            && padding[axis * 2 + 1] == total_padding - total_padding / 2
            && i64::from(output_shape[axis]) == same_size;
        valid &= padding[axis * 2..axis * 2 + 2] == [0, 0]
            && size >= filter
            && i64::from(output_shape[axis]) == (size - filter) / stride + 1;
    }
    let padding_mode = if valid {
        1
    } else if same && strides[1] == 1 && strides[2] == 1 {
        0
    } else {
        return None;
    };
    let body = subgraphs.table(options.u32(5)? as usize)?;
    let body_inputs = body.vector(1, 4)?.words()?;
    let body_outputs = body.vector(2, 4)?.words()?;
    let body_ops = body.vector(3, 4)?;
    if body_inputs.len() != 2 || body_outputs.len() != 1 || body_ops.len() != 1 {
        return None;
    }
    let body_tensors = body.vector(0, 4)?;
    for &index in body_inputs.iter().chain(&body_outputs) {
        let tensor = body_tensors.table(index as usize)?;
        if tensor.u8(1)? != FLOAT32 || !tensor.vector(0, 4)?.bytes.is_empty() {
            return None;
        }
    }
    let maximum = body_ops.table(0)?;
    if opcode(codes.table(maximum.u32(0)? as usize)?)? != MAXIMUM
        || maximum.vector(1, 4)?.words()? != body_inputs
        || maximum.vector(2, 4)?.words()? != body_outputs
    {
        return None;
    }
    Some(Pool {
        input: inputs[0],
        output: outputs[0],
        padding: padding_mode,
        height: window[1] as u32,
        width: window[2] as u32,
        stride_h: strides[1] as u32,
        stride_w: strides[2] as u32,
    })
}

fn constant_data<'a>(tensor: Table<'a>, buffers: Vector<'a>) -> Option<&'a [u8]> {
    buffers.table(tensor.u32(2)? as usize)?.vector(0, 1).map(|v| v.bytes)
}

fn opcode(code: Table<'_>) -> Option<u32> {
    Some(code.u32(3)?.max(u32::from(code.u8(0)?)))
}

fn append_pool(data: &mut Vec<u8>, base: usize, code: u32, pool: Pool) -> Option<usize> {
    let op = append_table(data, base, &[code, 0, 0, 5, 0])?;
    let options = append_table(
        data,
        base,
        &[pool.padding, pool.stride_w, pool.stride_h, pool.width, pool.height, 0],
    )?;
    let inputs = base.checked_add(data.len())?;
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&pool.input.to_le_bytes());
    let outputs = base.checked_add(data.len())?;
    data.extend_from_slice(&1u32.to_le_bytes());
    data.extend_from_slice(&pool.output.to_le_bytes());
    for (field, target) in [(op + 8, inputs), (op + 12, outputs), (op + 20, options)] {
        data[field - base..field - base + 4]
            .copy_from_slice(&u32::try_from(target.checked_sub(field)?).ok()?.to_le_bytes());
    }
    Some(op)
}

fn append_table(data: &mut Vec<u8>, base: usize, fields: &[u32]) -> Option<usize> {
    while (base + data.len()) % 4 != 0 {
        data.push(0);
    }
    let vtable = base.checked_add(data.len())?;
    data.extend_from_slice(&u16::try_from(4 + fields.len() * 2).ok()?.to_le_bytes());
    data.extend_from_slice(&u16::try_from(4 + fields.len() * 4).ok()?.to_le_bytes());
    for i in 0..fields.len() {
        data.extend_from_slice(&u16::try_from(4 + i * 4).ok()?.to_le_bytes());
    }
    while (base + data.len()) % 4 != 0 {
        data.push(0);
    }
    let table = base.checked_add(data.len())?;
    data.extend_from_slice(&i32::try_from(table.checked_sub(vtable)?).ok()?.to_le_bytes());
    for field in fields {
        data.extend_from_slice(&field.to_le_bytes());
    }
    Some(table)
}

#[derive(Clone, Copy)]
struct Table<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Table<'a> {
    fn field(self, slot: usize) -> Option<usize> {
        let distance = i32::from_le_bytes(read(self.data, self.position)?);
        let vtable = self.position.checked_add_signed(-(distance as isize))?;
        let length = usize::from(u16::from_le_bytes(read(self.data, vtable)?));
        let entry = 4usize.checked_add(slot.checked_mul(2)?)?;
        if entry.checked_add(2)? > length {
            return None;
        }
        let offset = usize::from(u16::from_le_bytes(read(self.data, vtable.checked_add(entry)?)?));
        if offset == 0 {
            return None;
        }
        let position = self.position.checked_add(offset)?;
        self.data.get(position)?;
        Some(position)
    }
    fn u32(self, slot: usize) -> Option<u32> {
        self.field(slot).map_or(Some(0), |p| word(self.data, p))
    }
    fn u8(self, slot: usize) -> Option<u8> {
        self.field(slot).map_or(Some(0), |p| self.data.get(p).copied())
    }
    fn table(self, slot: usize) -> Option<Self> {
        let field = self.field(slot)?;
        Some(Self {
            data: self.data,
            position: field.checked_add(word(self.data, field)? as usize)?,
        })
    }
    fn vector(self, slot: usize, width: usize) -> Option<Vector<'a>> {
        let Some(field) = self.field(slot) else {
            return Some(Vector { data: self.data, bytes: &[], position: 0, width });
        };
        let vector = field.checked_add(word(self.data, field)? as usize)?;
        let length = word(self.data, vector)? as usize;
        let position = vector.checked_add(4)?;
        let end = position.checked_add(length.checked_mul(width)?)?;
        Some(Vector { data: self.data, bytes: self.data.get(position..end)?, position, width })
    }
}

#[derive(Clone, Copy)]
struct Vector<'a> {
    data: &'a [u8],
    bytes: &'a [u8],
    position: usize,
    width: usize,
}
impl<'a> Vector<'a> {
    fn len(self) -> usize {
        self.bytes.len() / self.width
    }
    fn word(self, index: usize) -> Option<u32> {
        word(self.bytes, index.checked_mul(4)?)
    }
    fn words(self) -> Option<Vec<u32>> {
        (0..self.len()).map(|i| self.word(i)).collect()
    }
    fn longs(self) -> Option<Vec<i64>> {
        (0..self.len())
            .map(|i| Some(i64::from_le_bytes(read(self.bytes, i.checked_mul(8)?)?)))
            .collect()
    }
    fn table(self, index: usize) -> Option<Table<'a>> {
        let field = self.position.checked_add(index.checked_mul(4)?)?;
        Some(Table { data: self.data, position: field.checked_add(self.word(index)? as usize)? })
    }
}
fn read<const N: usize>(data: &[u8], position: usize) -> Option<[u8; N]> {
    data.get(position..position.checked_add(N)?)?.try_into().ok()
}
fn word(data: &[u8], position: usize) -> Option<u32> {
    Some(u32::from_le_bytes(read(data, position)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FP16: &[u8] = include_bytes!("../tests/fixtures/fp16_constant_partition.tflite");
    const POOL: &[u8] = include_bytes!("../tests/fixtures/stablehlo_max_pool.tflite");

    fn model(data: &[u8]) -> Table<'_> {
        Table { data, position: word(data, 0).unwrap() as usize }
    }

    #[test]
    fn constant_conversion_precedes_image_operations() {
        let mut data = FP16.to_vec();
        prepare(&mut data).unwrap();
        let model = model(&data);
        let operators = model.vector(2, 4).unwrap().table(0).unwrap().vector(3, 4).unwrap();
        let codes = model.vector(1, 4).unwrap();
        let actual: Vec<_> = (0..operators.len())
            .map(|i| {
                opcode(codes.table(operators.table(i).unwrap().u32(0).unwrap() as usize).unwrap())
                    .unwrap()
            })
            .collect();
        assert_eq!(actual, [DEQUANTIZE, 18, 18, 18, 53, 53]);
        assert_eq!(data.len(), FP16.len());
        let once = data.clone();
        prepare(&mut data).unwrap();
        assert_eq!(data, once);
    }

    #[test]
    fn runtime_conversion_keeps_its_order() {
        let mut data = FP16.to_vec();
        let tensor = model(&data)
            .vector(2, 4)
            .unwrap()
            .table(0)
            .unwrap()
            .vector(0, 4)
            .unwrap()
            .table(2)
            .unwrap();
        let buffer_index = tensor.field(2).unwrap();
        data[buffer_index..buffer_index + 4].copy_from_slice(&0u32.to_le_bytes());
        let original = data.clone();
        prepare(&mut data).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn supported_pooling_is_idempotent() {
        let mut data = POOL.to_vec();
        prepare(&mut data).unwrap();
        let model = model(&data);
        let operator =
            model.vector(2, 4).unwrap().table(0).unwrap().vector(3, 4).unwrap().table(1).unwrap();
        assert_eq!(
            opcode(model.vector(1, 4).unwrap().table(operator.u32(0).unwrap() as usize).unwrap()),
            Some(MAX_POOL_2D)
        );
        assert_eq!(operator.vector(1, 4).unwrap().words().unwrap(), [2]);
        assert_eq!(operator.u8(3), Some(5));
        assert_eq!(operator.table(4).unwrap().u32(3), Some(3));
        let once = data.clone();
        prepare(&mut data).unwrap();
        assert_eq!(data, once);
    }

    #[test]
    fn dilated_pooling_is_preserved() {
        let mut data = POOL.to_vec();
        let options = model(&data)
            .vector(2, 4)
            .unwrap()
            .table(0)
            .unwrap()
            .vector(3, 4)
            .unwrap()
            .table(1)
            .unwrap()
            .table(12)
            .unwrap();
        let dilation = options.vector(3, 8).unwrap().position + 8;
        data[dilation..dilation + 8].copy_from_slice(&2i64.to_le_bytes());
        let original = data.clone();
        prepare(&mut data).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn aliased_opcode_with_an_unsupported_use_is_preserved() {
        let mut data = POOL.to_vec();
        let codes = model(&data).vector(1, 4).unwrap();
        let reduction_code = codes.table(1).unwrap().position;
        let multiply_entry = codes.position + 3 * 4;
        data[multiply_entry..multiply_entry + 4]
            .copy_from_slice(&((reduction_code - multiply_entry) as u32).to_le_bytes());
        let original = data.clone();
        prepare(&mut data).unwrap();
        assert_eq!(data, original);
    }

    #[test]
    fn truncated_models_never_panic_or_partially_apply_edits() {
        for fixture in [FP16, POOL] {
            for end in 0..fixture.len() {
                let mut data = fixture[..end].to_vec();
                if prepare(&mut data).is_none() {
                    assert_eq!(data, fixture[..end]);
                }
            }
        }
    }

    #[test]
    fn malformed_offsets_never_panic_or_partially_apply_edits() {
        for fixture in [FP16, POOL] {
            for offset in (0..fixture.len()).step_by(7) {
                for value in [0, 0x80, 0xff] {
                    let mut data = fixture.to_vec();
                    data[offset] = value;
                    let original = data.clone();
                    if prepare(&mut data).is_none() {
                        assert_eq!(data, original);
                    }
                }
            }
        }
    }
}
