from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx

class SMILESToGraphMapper:
    def __init__(self):
        self.mol = None
        self.G = None
        self.atom_map = {}
        self.bond_map = {}

    def map_tokens_to_graph(self, smiles, tokens):
        # Create molecule from SMILES
        self.mol = Chem.MolFromSmiles(smiles)
        
        # Add explicit hydrogens
        self.mol = Chem.AddHs(self.mol)
        
        # Generate 2D coordinates (not necessary for mapping, but useful for visualization)
        AllChem.Compute2DCoords(self.mol)
        
        # Create graph representation
        self.G = nx.Graph()
        
        # Add nodes (atoms)
        for atom in self.mol.GetAtoms():
            self.G.add_node(atom.GetIdx(), 
                            element=atom.GetSymbol(), 
                            charge=atom.GetFormalCharge(),
                            implicit_hs=atom.GetNumImplicitHs())
        
        # Add edges (bonds)
        for bond in self.mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            self.G.add_edge(start, end, bond_type=bond_type)
        
        # Map tokens to atoms and bonds
        token_to_atom = {}
        token_to_bond = {}
        
        atom_idx = 0
        bond_idx = 0
        
        for i, token in enumerate(tokens):
            if token in ['(', ')', '=', '#']:  # These tokens don't directly correspond to atoms or bonds
                continue
            elif token == '[' or token == ']':  # Start or end of an atom block
                continue
            elif token.isalpha():  # Atom symbol
                token_to_atom[i] = atom_idx
                atom_idx += 1
            elif token == '-':  # Single bond
                token_to_bond[i] = bond_idx
                bond_idx += 1
        
        num_atoms = self.mol.GetNumAtoms()
        num_bonds = self.mol.GetNumBonds()
        
        return token_to_atom, token_to_bond, num_atoms, num_bonds

    def get_atom_info(self, atom_idx):
        atom = self.mol.GetAtomWithIdx(atom_idx)
        return {
            'element': atom.GetSymbol(),
            'charge': atom.GetFormalCharge(),
            'implicit_hs': atom.GetNumImplicitHs()
        }

    def get_bond_info(self, bond_idx):
        bond = self.mol.GetBondWithIdx(bond_idx)
        return {
            'start': bond.GetBeginAtomIdx(),
            'end': bond.GetEndAtomIdx(),
            'type': bond.GetBondTypeAsDouble()
        }

    def print_graph_info(self):
        print("\nVertices (format: index: element (charge, implicit hydrogens)):")
        for node, data in self.G.nodes(data=True):
            print(f"{node}: {data['element']} (charge: {data['charge']}, implicit H: {data['implicit_hs']})")
        
        print("\nEdges (with bond types):")
        for start, end, data in self.G.edges(data=True):
            print(f"{start} -- {end} (Bond type: {data['bond_type']})")

if __name__ == '__main__':
    # Example usage
    mapper = SMILESToGraphMapper()
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    tokens = list(smiles)  # Simple tokenization, you might want to use a more sophisticated method

    token_to_atom, token_to_bond, num_atoms, num_bonds = mapper.map_tokens_to_graph(smiles, tokens)

    print(f"Number of atoms: {num_atoms}")
    print(f"Number of bonds: {num_bonds}")
    print("\nToken to Atom mapping:")
    for token_idx, atom_idx in token_to_atom.items():
        print(f"Token {tokens[token_idx]} at position {token_idx} maps to atom {atom_idx}: {mapper.get_atom_info(atom_idx)}")
    print("\nToken to Bond mapping:")
    for token_idx, bond_idx in token_to_bond.items():
        print(f"Token {tokens[token_idx]} at position {token_idx} maps to bond {bond_idx}: {mapper.get_bond_info(bond_idx)}")

    mapper.print_graph_info()