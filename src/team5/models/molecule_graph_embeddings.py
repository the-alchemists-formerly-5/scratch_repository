import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem


class MoleculeEmbeddingModule(nn.Module):
    def __init__(self, tokenizer, embedding_model, embedding_dim):
        super(MoleculeEmbeddingModule, self).__init__()
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim

        # Random embedding generators
        self.random_atom_embedding = nn.Embedding(
            100, embedding_dim
        )  # 100 is an arbitrary max number of atom types
        self.random_bond_embedding = nn.Embedding(
            10, embedding_dim
        )  # 10 is an arbitrary max number of bond types

    def forward(self, smiles):
        # Tokenize original SMILES
        tokens = self.tokenizer.tokenize(smiles)

        # Get embeddings from the embedding model
        token_embeddings = self.embedding_model(tokens)

        # Process molecule structure
        mol = Chem.MolFromSmiles(smiles)
        mol_with_h = Chem.AddHs(mol)

        # Generate atom and bond tensors
        atom_tensors, bond_tensors = self._generate_embeddings(
            mol_with_h, tokens, token_embeddings, smiles
        )

        return atom_tensors, bond_tensors

    def _generate_embeddings(self, mol, tokens, token_embeddings, original_smiles):
        atom_embeddings = []
        bond_embeddings = []

        # Create a map of each atom to its position in the original SMILES string
        atom_to_smiles_map = {}
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1:  # Skip hydrogens
                idx = atom.GetIdx()
                env = list(AllChem.FindAtomEnvironmentOfRadiusN(mol, 0, idx))
                if env:
                    submol = Chem.PathToSubmol(mol, env)
                    subsmiles = Chem.MolToSmiles(submol)
                    smiles_idx = original_smiles.find(subsmiles)
                    if smiles_idx != -1:
                        atom_to_smiles_map[idx] = smiles_idx

        # Process atoms
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetAtomicNum() == 1:  # Hydrogen
                # Use a random embedding for hydrogens
                embedding = self.random_atom_embedding(torch.tensor(1))
            else:
                smiles_idx = atom_to_smiles_map.get(idx)
                if smiles_idx is not None:
                    # Find the corresponding token
                    token_idx = next(
                        (
                            i
                            for i, t in enumerate(tokens)
                            if t.startswith(atom.GetSymbol())
                            and sum(len(t) for t in tokens[:i])
                            <= smiles_idx
                            < sum(len(t) for t in tokens[: i + 1])
                        ),
                        None,
                    )
                    if token_idx is not None:
                        # Use the embedding from the model
                        embedding = token_embeddings[token_idx]
                    else:
                        # Use a random embedding if no matching token is found
                        embedding = self.random_atom_embedding(
                            torch.tensor(atom.GetAtomicNum())
                        )
                else:
                    # Use a random embedding for atoms not found in original SMILES
                    embedding = self.random_atom_embedding(
                        torch.tensor(atom.GetAtomicNum())
                    )

            atom_embeddings.append(embedding)

        # Process bonds
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # Find the corresponding token for this bond
            begin_smiles_idx = atom_to_smiles_map.get(begin_idx)
            end_smiles_idx = atom_to_smiles_map.get(end_idx)

            if begin_smiles_idx is not None and end_smiles_idx is not None:
                # The bond is represented by tokens between these two indices
                bond_smiles = original_smiles[begin_smiles_idx : end_smiles_idx + 1]
                bond_token = next(
                    (t for t in tokens if t in bond_smiles and t in ["=", "#", "-"]),
                    None,
                )

                if bond_token:
                    # Use the embedding from the model
                    embedding = token_embeddings[tokens.index(bond_token)]
                else:
                    # Use a random embedding if no specific bond token is found
                    embedding = self.random_bond_embedding(torch.tensor(int(bond_type)))
            else:
                # Use a random embedding if the bond isn't clearly represented in the SMILES
                embedding = self.random_bond_embedding(torch.tensor(int(bond_type)))

            bond_embeddings.append(embedding)

        return torch.stack(atom_embeddings), torch.stack(bond_embeddings)


# Example usage:
# tokenizer = SimpleSMILESTokenizer('seyonec/ChemBERTa-zinc-base-v1')
# embedding_model = ... # Your embedding model that takes tokens and returns embeddings
# embedding_dim = 256  # Set this to match your embedding model's output dimension
#
# module = MoleculeEmbeddingModule(tokenizer, embedding_model, embedding_dim)
#
# smiles = "CCO"  # Ethanol
# atom_embeddings, bond_embeddings = module(smiles)
