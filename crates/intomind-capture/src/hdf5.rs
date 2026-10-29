//! The part of HDF5 a MATLAB v7.3 file needs, written by hand.
//!
//! A MATLAB v7.3 file is an HDF5 file with a 512-byte user block whose first
//! 128 bytes are MATLAB's own header. An export holds one group, the root,
//! with at most eight variables, each a contiguous one- or two-dimensional
//! array with a `MATLAB_class` attribute and, for logical and char
//! variables, a `MATLAB_int_decode` one. This writes exactly that, in the
//! structures the HDF5 library writes for the same file by default: the
//! version 0 superblock, a symbol table group with one B-tree node, one
//! local heap and one symbol node, and version 1 object headers. No HDF5
//! library is needed to write it, and any HDF5 reader, MATLAB's included,
//! reads it.

/// The undefined address.
const UNDEF: u64 = u64::MAX;
/// The user block HDF5 data starts after, and where every address counts from.
const BASE: u64 = 512;
/// A group's leaf and internal node sizes, HDF5's defaults.
const LEAF_K: usize = 4;
const NODE_K: usize = 16;
/// The size HDF5 gives a dataset's object header, filled out with a null
/// message.
const OHDR_SIZE: usize = 256;

/// The element types a variable may have, as HDF5 describes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Elem {
    U8,
    U16,
    U32,
    U64,
    I16,
    I32,
    I64,
    F32,
    F64,
}

impl Elem {
    fn size(self) -> usize {
        match self {
            Elem::U8 => 1,
            Elem::U16 | Elem::I16 => 2,
            Elem::U32 | Elem::I32 | Elem::F32 => 4,
            Elem::U64 | Elem::I64 | Elem::F64 => 8,
        }
    }

    /// The datatype message's body.
    fn datatype(self) -> Vec<u8> {
        let mut d = Vec::new();
        let size = self.size() as u32;
        match self {
            Elem::F32 | Elem::F64 => {
                // Version 1, floating point, little endian, implied leading
                // one, the sign in the top bit.
                let (sign, exp_loc, exp_size, mant_size, bias): (u8, u8, u8, u8, u32) =
                    if self == Elem::F64 { (63, 52, 11, 52, 1023) } else { (31, 23, 8, 23, 127) };
                d.extend([0x11, 0x20, sign, 0x00]);
                d.extend(size.to_le_bytes());
                d.extend(0u16.to_le_bytes());
                d.extend(((size * 8) as u16).to_le_bytes());
                d.extend([exp_loc, exp_size, 0, mant_size]);
                d.extend(bias.to_le_bytes());
            }
            _ => {
                let signed = matches!(self, Elem::I16 | Elem::I32 | Elem::I64);
                d.extend([0x10, if signed { 0x08 } else { 0x00 }, 0x00, 0x00]);
                d.extend(size.to_le_bytes());
                d.extend(0u16.to_le_bytes());
                d.extend(((size * 8) as u16).to_le_bytes());
            }
        }
        d
    }
}

/// One variable: its name, its shape, its element type, its bytes in C
/// order, and its MATLAB attributes.
pub(crate) struct Variable {
    pub(crate) name: String,
    pub(crate) dims: Vec<u64>,
    pub(crate) elem: Elem,
    pub(crate) data: Vec<u8>,
    pub(crate) matlab_class: String,
    pub(crate) int_decode: Option<i64>,
}

fn pad8(v: &mut Vec<u8>) {
    while v.len() & 7 != 0 {
        v.push(0);
    }
}

fn align8(n: usize) -> usize {
    n.div_ceil(8) * 8
}

/// One message of a version 1 object header: its type, its flags, and its
/// body padded to eight bytes.
fn message(kind: u16, flags: u8, body: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(8 + align8(body.len()));
    m.extend(kind.to_le_bytes());
    m.extend((align8(body.len()) as u16).to_le_bytes());
    m.push(flags);
    m.extend([0, 0, 0]);
    m.extend(body);
    pad8(&mut m);
    m
}

/// A simple dataspace with its maximum dimensions equal to its dimensions,
/// as HDF5 writes one for a fixed-size array. No dimensions is a scalar.
fn dataspace(dims: &[u64]) -> Vec<u8> {
    let mut d = vec![1u8, dims.len() as u8, if dims.is_empty() { 0 } else { 1 }, 0, 0, 0, 0, 0];
    for &x in dims {
        d.extend(x.to_le_bytes());
    }
    if !dims.is_empty() {
        for &x in dims {
            d.extend(x.to_le_bytes());
        }
    }
    d
}

/// A version 1 attribute message.
fn attribute(name: &str, datatype: &[u8], value: &[u8]) -> Vec<u8> {
    let space = dataspace(&[]);
    let mut a = vec![1u8, 0];
    a.extend(((name.len() + 1) as u16).to_le_bytes());
    a.extend((datatype.len() as u16).to_le_bytes());
    a.extend((space.len() as u16).to_le_bytes());
    let mut n = name.as_bytes().to_vec();
    n.push(0);
    pad8(&mut n);
    a.extend(n);
    let mut t = datatype.to_vec();
    pad8(&mut t);
    a.extend(t);
    a.extend(space);
    a.extend(value);
    a
}

/// `MATLAB_class`, a fixed-length ASCII string, as h5py stores a numpy
/// bytes scalar.
fn class_attribute(class: &str) -> Vec<u8> {
    let mut t = vec![0x13u8, 0x01, 0x00, 0x00];
    t.extend((class.len() as u32).to_le_bytes());
    attribute("MATLAB_class", &t, class.as_bytes())
}

/// `MATLAB_int_decode`, a 64-bit integer.
fn decode_attribute(v: i64) -> Vec<u8> {
    attribute("MATLAB_int_decode", &Elem::I64.datatype(), &v.to_le_bytes())
}

/// A version 1 object header holding `messages`, filled out to the size the
/// HDF5 library gives a dataset's header with a null message.
fn object_header(messages: &[Vec<u8>]) -> Vec<u8> {
    let used: usize = messages.iter().map(Vec::len).sum();
    let mut all: Vec<Vec<u8>> = messages.to_vec();
    let size = if used + 8 <= OHDR_SIZE {
        all.push(message(0x0000, 0, &vec![0u8; OHDR_SIZE - used - 8]));
        OHDR_SIZE
    } else {
        used
    };
    let mut h = vec![1u8, 0];
    h.extend((all.len() as u16).to_le_bytes());
    h.extend(1u32.to_le_bytes());
    h.extend((size as u32).to_le_bytes());
    h.extend([0, 0, 0, 0]);
    for m in all {
        h.extend(m);
    }
    h
}

/// The file: the user block with `header` at its start, then the HDF5 data.
pub(crate) fn write(header: &[u8; 128], vars: &[Variable]) -> Vec<u8> {
    assert!(vars.len() <= 2 * LEAF_K, "one symbol node holds eight variables");
    // Where everything goes, as addresses from the base.
    let superblock_len = 96;
    let root_at = superblock_len as u64;
    let root_len = 16 + 24;
    let btree_at = root_at + root_len as u64;
    let btree_len = 24 + (2 * NODE_K + 1) * 8 + 2 * NODE_K * 8;
    let heap_at = btree_at + btree_len as u64;
    // The heap: the empty name at offset zero, then each name padded to eight.
    let mut heap_data = vec![0u8; 8];
    let mut name_at = Vec::with_capacity(vars.len());
    for v in vars {
        name_at.push(heap_data.len() as u64);
        heap_data.extend(v.name.as_bytes());
        heap_data.push(0);
        pad8(&mut heap_data);
    }
    let heap_data_at = heap_at + 32;
    let snod_at = heap_data_at + heap_data.len() as u64;
    let snod_len = 8 + 2 * LEAF_K * 40;
    let mut next = snod_at + snod_len as u64;

    // Each variable's object header, and its data right after it.
    let mut objects: Vec<(u64, Vec<u8>, Vec<u8>)> = Vec::new();
    for v in vars {
        let at = next;
        let data_len = v.data.len() as u64;
        // The header's size is known before its data's address.
        let hdr_len = {
            let probe = object_header(&messages_for(v, 0, data_len));
            probe.len() as u64
        };
        let data_at = if data_len == 0 { UNDEF } else { align8_u64(at + hdr_len) };
        let hdr = object_header(&messages_for(v, data_at, data_len));
        next = if data_len == 0 { at + hdr_len } else { data_at + data_len };
        next = align8_u64(next);
        objects.push((at, hdr, v.data.clone()));
    }
    let eof = BASE + next;

    let mut out = Vec::with_capacity(eof as usize);
    out.extend(header);
    out.resize(BASE as usize, 0);
    // Superblock, version 0.
    out.extend(b"\x89HDF\r\n\x1a\n");
    out.extend([0, 0, 0, 0, 0, 8, 8, 0]);
    out.extend((LEAF_K as u16).to_le_bytes());
    out.extend((NODE_K as u16).to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(BASE.to_le_bytes());
    out.extend(UNDEF.to_le_bytes());
    out.extend(eof.to_le_bytes());
    out.extend(UNDEF.to_le_bytes());
    // The root group's symbol table entry: cached B-tree and heap.
    out.extend(0u64.to_le_bytes());
    out.extend(root_at.to_le_bytes());
    out.extend(1u32.to_le_bytes());
    out.extend(0u32.to_le_bytes());
    out.extend(btree_at.to_le_bytes());
    out.extend(heap_at.to_le_bytes());
    // The root group's object header: one symbol table message.
    let mut st = Vec::new();
    st.extend(btree_at.to_le_bytes());
    st.extend(heap_at.to_le_bytes());
    let root = {
        let m = message(0x0011, 0, &st);
        let mut h = vec![1u8, 0];
        h.extend(1u16.to_le_bytes());
        h.extend(1u32.to_le_bytes());
        h.extend((m.len() as u32).to_le_bytes());
        h.extend([0, 0, 0, 0]);
        h.extend(m);
        h
    };
    out.extend(root);
    // The B-tree: one leaf entry pointing at the one symbol node, keyed by
    // the empty name and the largest name.
    let largest = (0..vars.len()).max_by(|&a, &b| vars[a].name.cmp(&vars[b].name)).map(|i| name_at[i]).unwrap_or(0);
    let mut bt = Vec::with_capacity(btree_len);
    bt.extend(b"TREE");
    bt.extend([0u8, 0]);
    bt.extend((if vars.is_empty() { 0u16 } else { 1 }).to_le_bytes());
    bt.extend(UNDEF.to_le_bytes());
    bt.extend(UNDEF.to_le_bytes());
    bt.extend(0u64.to_le_bytes());
    bt.extend(snod_at.to_le_bytes());
    bt.extend(largest.to_le_bytes());
    bt.resize(btree_len, 0);
    out.extend(bt);
    // The local heap, with no free space.
    out.extend(b"HEAP");
    out.extend([0u8, 0, 0, 0]);
    out.extend((heap_data.len() as u64).to_le_bytes());
    out.extend(1u64.to_le_bytes());
    out.extend(heap_data_at.to_le_bytes());
    out.extend(&heap_data);
    // The symbol node: an entry per variable, in name order.
    let mut order: Vec<usize> = (0..vars.len()).collect();
    order.sort_by(|&a, &b| vars[a].name.cmp(&vars[b].name));
    let mut sn = Vec::with_capacity(snod_len);
    sn.extend(b"SNOD");
    sn.extend([1u8, 0]);
    sn.extend((vars.len() as u16).to_le_bytes());
    for i in order {
        sn.extend(name_at[i].to_le_bytes());
        sn.extend(objects[i].0.to_le_bytes());
        sn.extend(0u32.to_le_bytes());
        sn.extend(0u32.to_le_bytes());
        sn.extend([0u8; 16]);
    }
    sn.resize(snod_len, 0);
    out.extend(sn);
    // The variables.
    for (at, hdr, data) in &objects {
        out.resize((BASE + at) as usize, 0);
        out.extend(hdr);
        if !data.is_empty() {
            let data_at = align8_u64(at + hdr.len() as u64);
            out.resize((BASE + data_at) as usize, 0);
            out.extend(data);
        }
    }
    out.resize(eof as usize, 0);
    out
}

fn align8_u64(n: u64) -> u64 {
    n.div_ceil(8) * 8
}

/// A dataset's messages, in the order the HDF5 library writes them.
fn messages_for(v: &Variable, data_at: u64, data_len: u64) -> Vec<Vec<u8>> {
    let mut layout = vec![3u8, 1];
    layout.extend(data_at.to_le_bytes());
    layout.extend(data_len.to_le_bytes());
    let mut m = vec![
        message(0x0001, 0, &dataspace(&v.dims)),
        message(0x0003, 1, &v.elem.datatype()),
        message(0x0005, 1, &[2, 2, 2, 1, 0, 0, 0, 0]),
        message(0x0008, 0, &layout),
        message(0x000C, 0, &class_attribute(&v.matlab_class)),
    ];
    if let Some(d) = v.int_decode {
        m.push(message(0x000C, 0, &decode_attribute(d)));
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_double_variable_has_the_header_the_hdf5_library_writes() {
        // The bytes h5py 3.16 with HDF5 2.0 writes for a (4, 2592) float64
        // dataset's datatype, fill value and MATLAB_class messages.
        assert_eq!(
            Elem::F64.datatype(),
            [0x11, 0x20, 0x3f, 0, 8, 0, 0, 0, 0, 0, 0x40, 0, 0x34, 0x0b, 0, 0x34, 0xff, 0x03, 0, 0]
        );
        assert_eq!(Elem::I64.datatype(), [0x10, 0x08, 0, 0, 8, 0, 0, 0, 0, 0, 0x40, 0]);
        let a = class_attribute("int64");
        assert_eq!(message(0x000C, 0, &a).len(), 8 + 48, "the size h5py's MATLAB_class message has");
        assert_eq!(&a[..8], &[1, 0, 13, 0, 8, 0, 8, 0]);
        assert_eq!(message(0x000C, 0, &decode_attribute(1)).len(), 8 + 64, "and its MATLAB_int_decode message");
        let v = Variable {
            name: "uv".into(),
            dims: vec![4, 2592],
            elem: Elem::F64,
            data: vec![0; 4 * 2592 * 8],
            matlab_class: "double".into(),
            int_decode: None,
        };
        let h = object_header(&messages_for(&v, 2048, 82944));
        assert_eq!(h.len(), 16 + OHDR_SIZE);
        assert_eq!(&h[..12], &[1, 0, 6, 0, 1, 0, 0, 0, 0, 1, 0, 0]);
    }

    #[test]
    fn a_file_starts_with_the_user_block_and_the_superblock() {
        let v = vec![Variable {
            name: "x".into(),
            dims: vec![1, 3],
            elem: Elem::U8,
            data: vec![1, 0, 1],
            matlab_class: "logical".into(),
            int_decode: Some(1),
        }];
        let f = write(&[b' '; 128], &v);
        assert_eq!(&f[512..520], b"\x89HDF\r\n\x1a\n");
        assert_eq!(u64::from_le_bytes(f[512 + 40..512 + 48].try_into().unwrap()), f.len() as u64);
    }
}
