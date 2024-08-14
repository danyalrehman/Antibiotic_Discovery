import rdkit
import sys
import csv
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from RAscore import RAscore_XGB
from multiprocessing import Pool
from rdkit import Chem
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import rdkit.Chem.QED as QED


def func(line):
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    catalog = FilterCatalog(params)
    smiles = line.strip("\r\n ").split(',')[1]
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        entry = catalog.GetFirstMatch(mol)
        return line.strip("\r\n ") if entry is None else None
    else:
        return None


def pains_brenk_filter():
    header = sys.stdin.readline()
    print(header.strip("\r\n "))
    with Pool(80) as pool:
        data = pool.map(func, sys.stdin.readlines())
    for line in data:
        if line is not None:
            print(line)


def ra_filter():
    xgb_scorer = RAscore_XGB.RAScorerXGB(model_path='./RAscore/models/XGB_gdbmedchem_ecfp_counts/model.pkl')
    header = sys.stdin.readline()
    print(header.strip("\r\n ") + ",RAscore")

    lines = sys.stdin.readlines()
    for line in tqdm(lines):
        items = line.strip("\r\n ").split(',')
        rat, smiles = items[:2]
        ra_score = xgb_scorer.predict(smiles)
        items.append(str(ra_score))
        if ra_score > 0.8:
            print(','.join(items))


def buyable_filter():
    buyable = {}
    with open("ttt") as f:
        for line in f:
            item = line.split('\t')
            buyable[item[0]] = Chem.MolFromSmiles(item[-1])

    sys.stdin.readline()
    for line in tqdm(sys.stdin.readlines()):
        amol = Chem.MolFromSmiles(line.split(',')[1])
        for name, bmol in buyable.items():
            if similarity(amol, bmol) >= 0.6:
                print(line, name, Chem.MolToSmiles(bmol))


def similarity(amol, bmol):
    fp1 = AllChem.GetMorganFingerprintAsBitVect(amol, 2, nBits=2048, useChirality=False)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(bmol, 2, nBits=2048, useChirality=False)
    return DataStructs.TanimotoSimilarity(fp1, fp2) 


def sim_filter():
    with open("data/existing_ab.txt") as f:
        ab = [Chem.MolFromSmiles(line) for line in f]

    header = sys.stdin.readline()
    print(header.strip("\r\n ") + ",Similarity")

    lines = sys.stdin.readlines()
    for line in tqdm(lines):
        item = line.strip("\r\n ").split(',')
        mol = Chem.MolFromSmiles(item[1])
        sim = max([similarity(mol, x) for x in ab])
        if sim < 0.4:
            print(','.join(item) + f",{sim:.4f}")


def ring_filter():
    print(sys.stdin.readline().strip("\r\n "))
    pat = Chem.MolFromSmarts('[x4]')
    for line in sys.stdin:
        mol = Chem.MolFromSmiles(line.split(',')[1])
        if not mol.HasSubstructMatch(pat):
            print(line.strip("\r\n "))

def qed_filter():
    print(sys.stdin.readline().strip("\r\n "))
    for line in sys.stdin:
        mol = Chem.MolFromSmiles(line.split(',')[1])
        if QED.qed(mol) > 0.5:
            print(line.strip("\r\n "))

def sub_filter():
    print(sys.stdin.readline().strip("\r\n "))
    pat = Chem.MolFromSmiles('C1NCC1')
    for line in sys.stdin:
        mol = Chem.MolFromSmiles(line.split(',')[1])
        if not mol.HasSubstructMatch(pat):
            print(line.strip("\r\n "))


if __name__ == "__main__":
    sim_filter()
