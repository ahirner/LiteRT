// Copyright 2026 Google LLC.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

use std::collections::HashSet;

// Field positions and enum values follow tflite/converter/schema/schema.fbs.
const DEQUANTIZE: u32 = 6;
const FLOAT32: u8 = 0;
const FLOAT16: u8 = 1;

pub(crate) fn prepare(data: &mut Vec<u8>) -> Option<()> {
    let plan = plan(data)?;
    for (position, value) in plan.words {
        data[position..position + 4].copy_from_slice(&value.to_le_bytes());
    }
    Some(())
}

#[derive(Default)]
struct Plan {
    words: Vec<(usize, u32)>,
}

struct Graph {
    entries: usize,
    operators: Vec<usize>,
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
        }
        constant_ops.extend(remaining_ops);
        graphs.push(Graph { entries: operators.position, operators: constant_ops });
    }

    let mut plan = Plan::default();
    for graph in graphs {
        for (index, position) in graph.operators.into_iter().enumerate() {
            let entry = graph.entries.checked_add(index.checked_mul(4)?)?;
            let relative = u32::try_from(position.checked_sub(entry)?).ok()?;
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

fn constant_data<'a>(tensor: Table<'a>, buffers: Vector<'a>) -> Option<&'a [u8]> {
    buffers.table(tensor.u32(2)? as usize)?.vector(0, 1).map(|v| v.bytes)
}

fn opcode(code: Table<'_>) -> Option<u32> {
    Some(code.u32(3)?.max(u32::from(code.u8(0)?)))
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
    fn truncated_models_never_panic_or_partially_apply_edits() {
        for fixture in [FP16] {
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
        for fixture in [FP16] {
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
