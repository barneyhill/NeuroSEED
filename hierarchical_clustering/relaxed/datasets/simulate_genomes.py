import msprime
import numpy as np
import sgkit as sg
import tskit

def simulate_ts(
    sample_size: int,
    length: int = 100,
    mutation_rate: float = 0.05,
    random_seed: int = 42,
) -> tskit.TreeSequence:
    """
    Simulate some data using msprime with recombination and mutation and
    return the resulting tskit TreeSequence.
    Note this method currently simulates with ploidy=1 to minimise the
    update from an older version. We should update to simulate data under
    a range of ploidy values.
    """
    ancestry_ts = msprime.sim_ancestry(
        sample_size,
        ploidy=2,
        recombination_rate=0.01,
        sequence_length=length,
        random_seed=random_seed,
    )

    # Make sure we generate some data that's not all from the same tree
    assert ancestry_ts.num_trees > 1
    return msprime.sim_mutations(
        ancestry_ts, rate=mutation_rate, random_seed=random_seed, model="binary",
    )

def ts_to_dataset(ts, chunks=None, samples=None, phased=False):
    """
    Convert the specified tskit tree sequence into an sgkit dataset.
    Note this just generates haploids for now - see the note above
    in simulate_ts.
    """
    if samples is None:
        samples = ts.samples()
    tables = ts.dump_tables()
    alleles = []
    genotypes = []
    max_alleles = 0

    for var in ts.variants(samples=samples):
        var_genotypes = var.genotypes
        if sum(var.genotypes) >= len(samples) / 2:  # Correct major/minor allele
            var_genotypes = 1 - var.genotypes

        alleles.append(var.alleles)
        max_alleles = max(max_alleles, len(var.alleles))
        genotypes.append(var_genotypes)
   
    padded_alleles = [
        list(site_alleles) + [""] * (max_alleles - len(site_alleles))
        for site_alleles in alleles
    ]
    alleles: sg.typing.ArrayLike = np.array(padded_alleles).astype("S")

    n_individuals = int(len(samples)/2)
    n_sites = len(tables.sites)
    
    genotypes = np.expand_dims(genotypes, axis=2).reshape((-1, n_individuals, 2))

    phase = np.full((n_sites, n_individuals), phased)

    ds = sg.create_genotype_call_dataset(
        variant_contig_names=["1"],
        variant_contig=np.zeros(len(tables.sites), dtype=int),
        variant_position=tables.sites.position.astype(int),
        variant_allele=alleles,
        sample_id=np.array([f"ind_{u}" for u in samples]).astype("U")[0:n_individuals],
        call_genotype=genotypes,
        call_genotype_phased=phase,
    )
    if chunks is not None:
        ds = ds.chunk(dict(zip(["variants", "samples"], chunks)))
    return ds