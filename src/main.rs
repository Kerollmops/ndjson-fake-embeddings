use std::collections::HashMap;
use std::io::{self, BufRead as _, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::mem;
use std::path::Path;

use bytemuck::{AnyBitPattern, PodCastError};
use fs_err::File;
use memmap2::Mmap;
use mimalloc::MiMalloc;
use rand::Rng;
use serde::Serialize;
use serde_json::ser::{CompactFormatter, Formatter};
use serde_json::value::RawValue;
use serde_json::{json, Serializer};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

fn main() -> anyhow::Result<()> {
    let embeddings = MatLEView::<f32>::new("comment-embs-data.mat", 512)?;
    let reader = BufReader::new(std::io::stdin());
    let mut writer = BufWriter::new(std::io::stdout());

    let mut buffer = Vec::new();
    let mut rng = rand::thread_rng();
    for (result, embedding) in reader.lines().zip(embeddings.iter().cycle()) {
        let line = result?;
        let mut document: HashMap<String, &RawValue> = serde_json::from_str(&line)?;

        // Alterate a bit every embedding
        let alt = rng.gen_range(-0.1..=0.1);
        let embedding: Vec<f32> = embedding.iter().map(|x| x + alt).collect();

        buffer.clear();
        let mut embser = Serializer::with_formatter(&mut buffer, SmallFloatFormatter::new());
        let embeddings = json!({ "default": embedding });
        embeddings.serialize(&mut embser)?;
        let embedding = serde_json::from_slice(&buffer)?;

        document.insert("_vector".to_owned(), embedding);
        serde_json::to_writer(&mut writer, &document)?;
    }

    writer.flush()?;

    Ok(())
}

pub struct MatLEView<T> {
    mmap: Mmap,
    dimensions: usize,
    _marker: PhantomData<T>,
}

impl<T: AnyBitPattern> MatLEView<T> {
    pub fn new(path: impl AsRef<Path>, dimensions: usize) -> io::Result<MatLEView<T>> {
        let file = File::open(path.as_ref())?;
        let mmap = unsafe { Mmap::map(&file)? };
        assert!((mmap.len() / mem::size_of::<T>()) % dimensions == 0);
        Ok(MatLEView { mmap, dimensions, _marker: PhantomData })
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        (self.mmap.len() / mem::size_of::<T>()) / self.dimensions
    }

    pub fn get(&self, index: usize) -> Option<Result<&[T], PodCastError>> {
        let tsize = mem::size_of::<T>();
        if (index * self.dimensions + self.dimensions) * tsize < self.mmap.len() {
            let start = index * self.dimensions;
            let bytes = &self.mmap[start * tsize..(start + self.dimensions) * tsize];
            match bytemuck::try_cast_slice::<u8, T>(bytes) {
                Ok(slice) => Some(Ok(slice)),
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &[T]> + Clone {
        (0..self.len()).map(|i| self.get(i).unwrap().unwrap())
    }

    pub fn get_all(&self) -> Vec<&[T]> {
        self.iter().collect()
    }
}

struct SmallFloatFormatter(CompactFormatter);

impl SmallFloatFormatter {
    pub fn new() -> SmallFloatFormatter {
        SmallFloatFormatter(CompactFormatter)
    }
}

impl Formatter for SmallFloatFormatter {
    fn write_f32<W>(&mut self, writer: &mut W, value: f32) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        write!(writer, "{value:.4}")
    }

    fn write_f64<W>(&mut self, writer: &mut W, value: f64) -> io::Result<()>
    where
        W: ?Sized + io::Write,
    {
        write!(writer, "{value:.4}")
    }
}
