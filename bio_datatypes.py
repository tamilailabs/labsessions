# =====================================================
# Python Core Data Types with Biopython Mapping
# =====================================================

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import CodonTable

#https://biopython.org/

print("\n========== NUMERIC DATA TYPES ==========")

# int
genome_length = 3200000000
print("Genome length:", genome_length, type(genome_length))

# float
gc_content = 41.2
print("GC content:", gc_content, type(gc_content))

# complex (rare but valid in scientific computing)
signal = 2 + 3j
print("Complex signal:", signal, type(signal))


print("\n========== TEXT / STRING DATA TYPE ==========")

gene_name = "BRCA1"
print("Gene name:", gene_name, type(gene_name))

# Biopython mapping: string → Seq
dna_string = "ATGCGTAC"
dna_seq = Seq(dna_string)
print("Biopython Seq:", dna_seq, type(dna_seq))


print("\n========== BOOLEAN DATA TYPE ==========")

is_coding = True
print("Is coding gene:", is_coding, type(is_coding))

print("Length > 1000:", genome_length > 1000)


print("\n========== SEQUENCE DATA TYPES ==========")

# list (mutable)
exons = ["exon1", "exon2", "exon3"]
exons.append("exon4")
print("Exons list:", exons, type(exons))

# tuple (immutable)
chromosome_location = (17, 43044295, 43125482)
print("Chromosome location:", chromosome_location, type(chromosome_location))

# range
positions = range(1, 6)
print("Genome positions:", list(positions), type(positions))

# string as sequence
print("First base of gene:", gene_name[0])


print("\n========== SET DATA TYPES ==========")

# set (unique values)
unique_codons = {"ATG", "TAA", "TAG", "ATG"}
print("Unique codons:", unique_codons, type(unique_codons))

# frozenset (immutable set)
essential_codons = frozenset({"ATG", "TGG"})
print("Essential codons:", essential_codons, type(essential_codons))


print("\n========== MAPPING DATA TYPE ==========")

# dict
gene_info = {
    "name": "BRCA1",
    "chromosome": 17,
    "coding": True
}
print("Gene info:", gene_info, type(gene_info))

# Biopython mapping: dictionary inside SeqRecord
record = SeqRecord(
    dna_seq,
    id="BRCA1",
    description="Example gene"
)
record.annotations["organism"] = "Homo sapiens"
print("SeqRecord annotations:", record.annotations)


print("\n========== BINARY DATA TYPES ==========")

# bytes
binary_sequence = b"ATGC"
print("Bytes:", binary_sequence, type(binary_sequence))

# bytearray (mutable)
mutable_binary = bytearray(b"ATGC")
mutable_binary[0] = 71  # ASCII for 'G'
print("Bytearray:", mutable_binary, type(mutable_binary))

# memoryview
mem_view = memoryview(binary_sequence)
print("MemoryView:", mem_view, type(mem_view))


print("\n========== NONE DATA TYPE ==========")

unknown_function = None
print("Unknown value:", unknown_function, type(unknown_function))


print("\n========== BIOPYTHON-SPECIFIC CORE OBJECTS ==========")

# Seq (immutable biological sequence)
print("Seq object:", dna_seq, type(dna_seq))

# SeqRecord (sequence + metadata)
print("SeqRecord object:", record, type(record))

# Codon table (dict-like)
standard_table = CodonTable.unambiguous_dna_by_name["Standard"]
print("Codon AAA codes for:", standard_table.forward_table["AAA"])
print("Codon table type:", type(standard_table.forward_table))


print("\n✅ ALL PYTHON CORE DATA TYPES ADDRESSED WITH BIOPYTHON CONTEXT")
