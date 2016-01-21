import pandas as pd
import numbers
import numpy as np
import networkx as nx

from scipy.spatial.distance import pdist, squareform

# A list of backbone chain atoms
BACKBONE_ATOMS = ['N', 'CA', 'C', 'O']

class AtomicInteractionNetwork(object):
	"""docstring for AtomicInteractionNetwork"""
	def __init__(self, pdb_handle):
		super(AtomicInteractionNetwork, self).__init__()
		self.pdb_handle = pdb_handle
		self.dataframe = self.parse_pdb()
		self.distmat = self.compute_distmat(self.dataframe)
		self.rgroup_df = self.get_rgroup_dataframe()
		self.masterG = nx.Graph()

	def compute_interaction_graph(self):
		"""
		Computes the interaction graph in an automated fashion. 

		Graph definition and metadata:
		==============================
		- Node: Amino acid position.
			- aa: amino acid identity

		- Edge: Any interaction found by the atomic interaction network.
			- hbond: 			BOOLEAN 
			- disulfide: 		BOOLEAN 
			- hydrophobic: 		BOOLEAN 
			- ionic: 			BOOLEAN 
			- aromatic: 		BOOLEAN 
			- aromatic_sulphur: BOOLEAN
			- cation_pi: 		BOOLEAN 
		"""
		# Populate nodes, which are amino acid positions, and have metadata 
		nums_and_names = set(zip(self.dataframe['resi_num'], self.dataframe['resi_name']))
		for num, name in nums_and_names:
			self.masterG.add_node(num, aa=name)

		funcs = dict()
		funcs['hydrophobic'] = self.get_hydrophobic_interactions
		funcs['disulfide'] = self.get_disulfide_interactions
		funcs['hbond'] = self.get_hydrophobic_interactions
		funcs['ionic'] = self.get_ionic_interactions
		funcs['aromatic'] = self.get_aromatic_interactions
		funcs['aromatic_sulphur'] = self.get_aromatic_sulphur_interactions
		funcs['cation_pi'] = self.get_cation_pi_interactions

		# Add in each type of edge.
		for k, v in funcs.items():
			for r1, r2 in v():
				if (r1, r2) not in self.masterG.edges():
					attrs = dict()
					attrs[k] = True
					self.masterG.add_edge(r1, r2, attr_dict=attrs)
				else:
					self.masterG.edge[r1][r2][k] = True


	def parse_pdb(self):
		"""
		Parses the PDB file as a pandas DataFrame object.

		Backbone chain atoms are ignored for the calculation
		of interacting residues.
		"""
		atomic_data = []
		with open(self.pdb_handle, 'r') as f:
			for line in f.readlines():
				data = dict()
				if line[0:4] == 'ATOM':
					
					data['Record name'] = line[0:5].strip(' ')
					data['serial_number'] = int(line[6:11].strip(' '))
					data['atom'] = line[12:15].strip(' ')
					data['resi_name'] = line[17:20]
					data['chain_id'] = line[21]
					data['resi_num'] = int(line[23:26])
					data['x'] = float(line[30:37])
					data['y'] = float(line[38:45])
					data['z'] = float(line[46:53])

					atomic_data.append(data)
		atomic_df = pd.DataFrame(atomic_data)
		return atomic_df

	def compute_distmat(self, dataframe):
		"""
		Computes the pairwise euclidean distances between every atom.

		Design choice: passed in a DataFrame to enable easier testing on 
		dummy data. 
		"""

		euclidean_distances = pdist(dataframe[['x', 'y', 'z']], \
			metric='euclidean')

		return squareform(euclidean_distances)
		
	def get_interacting_atoms(self, angstroms, distmat):
		"""
		Finds the atoms that are within a particular radius of one another.
		"""
		return np.where(distmat <= angstroms)

	def get_interacting_resis(self, interacting_atoms, dataframe):
		"""
		Returns a list of 2-tuples indicating the interacting 
		residues based on the interacting atoms.

		Also filters out the list such that the residues have to be at least 
		two apart.
		"""
		resi1 = dataframe.ix[interacting_atoms[0]]['resi_num'].values
		resi2 = dataframe.ix[interacting_atoms[1]]['resi_num'].values

		interacting_resis = set(list(zip(resi1, resi2)))
		filtered_interacting_resis = set()
		for i1, i2 in interacting_resis:
			if abs(i1 - i2) >= 2:
				filtered_interacting_resis.add((i1, i2))

		return filtered_interacting_resis


	def get_rgroup_dataframe(self):
		"""
		Returns just the atoms that are amongst the R-groups and not part of 
		the backbone chain.
		"""

		rgroup_df = self.filter_dataframe(self.dataframe, 'atom', BACKBONE_ATOMS, False)
		return rgroup_df

	def filter_dataframe(self, dataframe, by_column, list_of_values, boolean):
		"""
		Filters the [dataframe] such that the [by_column] values have to be
		in the [list_of_values] list, if boolean == True, or not in the list if 
		boolean == False 
		"""
		df = dataframe.copy()
		df = df[df[by_column].isin(list_of_values) == boolean]
		df.reset_index(inplace=True, drop=True)

		return df

	##### SPECIFIC INTERACTION FUNCTIONS #####
	def get_hydrophobic_interactions(self):
		"""
		Finds all hydrophobic interactions between the following residues:
		ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO, TYR

		Criteria: R-group residues are within 5A distance.
		"""
		HYDROPHOBIC_RESIS = ['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE',
							 'TRP', 'PRO', 'TYR']

		hydrophobics_df = self.filter_dataframe(self.rgroup_df, 'resi_name', HYDROPHOBIC_RESIS, True)
		distmat = self.compute_distmat(hydrophobics_df)
		interacting_atoms = self.get_interacting_atoms(5, distmat)
		interacting_resis = self.get_interacting_resis(interacting_atoms, hydrophobics_df)

		return interacting_resis

	def get_disulfide_interactions(self):
		"""
		Finds all disulfide interactions between CYS residues, such that the 
		sulfur atom pairs are within 2.2A of each other.
		"""
		DISULFIDE_RESIS = ['CYS']
		DISULFIDE_ATOMS = ['SG']

		disulfide_df = self.filter_dataframe(self.rgroup_df, 'resi_name', DISULFIDE_RESIS, True)
		disulfide_df = self.filter_dataframe(disulfide_df, 'atom', DISULFIDE_ATOMS, True)
		distmat = self.compute_distmat(disulfide_df)
		interacting_atoms = self.get_interacting_atoms(2.2, distmat)
		interacting_resis = self.get_interacting_resis(interacting_atoms, disulfide_df)

		return interacting_resis


	def get_hydrogen_bond_interactions(self):
		"""
		Finds all hydrogen-bond interactions between atoms capable of hydrogen 
		bonding.
		"""

		def get_interacting_residues(HBOND_ATOMS, distance):
			hbond_df = self.filter_dataframe(self.rgroup_df, 'atom', HBOND_ATOMS, True)
			distmat = self.compute_distmat(hbond_df)
			# Find the interacting atoms for those that are within 3.5A of one another.
			interacting_atoms = self.get_interacting_atoms(distance, distmat)
			interacting_resis = self.get_interacting_resis(interacting_atoms, hbond_df)
			return interacting_resis
		# Double-check that this is true for all atoms
		HBOND_ATOMS = ['ND', 'NE', 'NH', 'NZ', 'OD', 'OE', 'OG', 'OH', 'SD', 'SG', 'N', 'O']
		interacting_resis = get_interacting_residues(HBOND_ATOMS, 3.5)

		# # The 3.5A criteria fits for all interactions involving the HBOND_ATOMS above.
		# # Now, just filter for those with SD and SG, to look for 4A interactions.
		HBOND_ATOMS_SULPHUR = ['SD', 'SG']
		interacting_resis = interacting_resis.union(get_interacting_residues(HBOND_ATOMS_SULPHUR, 4.0))
		return interacting_resis


	def get_ionic_interactions(self):
		"""
		Finds all ionic interactiosn between ARG, LYS, HIS, ASP, and GLU. 
		Distance cutoff: 6A.
		"""

		IONIC_RESIS = ['ARG', 'LYS', 'HIS', 'ASP', 'GLU']

		ionic_df = self.filter_dataframe(self.rgroup_df, 'resi_name', IONIC_RESIS, True)
		distmat = self.compute_distmat(ionic_df)
		interacting_atoms = self.get_interacting_atoms(6, distmat)
		interacting_resis = self.get_interacting_resis(interacting_atoms, ionic_df)
		return interacting_resis

	def get_aromatic_interactions(self):
		"""
		Finds all aromatic-aromatic interactions by looking for phenyl ring 
		centroids separated between 4.5A to 7A. 

		Phenyl rings are present on PHE, TRP, HIS and TYR. 

		Phenyl ring atoms on these amino acids are defined by the following 
		atoms:
		- PHE: CG, CD, CE, CZ
		- TRP: CD, CE, CH, CZ
		- HIS: CG, CD, ND, NE, CE 
		- TYR: CG, CD, CE, CZ

		Centroids of these atoms are taken by taking:
			(mean x), (mean y), (mean z)
		for each of the ring atoms.
		"""
		AROMATIC_RESIS = ['PHE', 'TRP', 'HIS', 'TYR']
		dfs = []
		for resi in AROMATIC_RESIS:
			resi_rings_df = self.get_ring_atoms(self.dataframe, resi)
			resi_centroid_df = self.get_ring_centroids(resi_rings_df, resi)
			dfs.append(resi_centroid_df)

		aromatic_df = pd.concat(dfs)
		aromatic_df.sort('resi_num', inplace=True)
		aromatic_df.reset_index(inplace=True, drop=True)

		distmat = self.compute_distmat(aromatic_df)
		interacting_atoms = np.where([(distmat >= 4.5) & (distmat <= 7)])
		interacting_resis = self.get_interacting_resis(interacting_atoms, aromatic_df)

		return interacting_resis

	# Helper functions for get_aromatic_interactions
	def get_ring_atoms(self, dataframe, aa):
		"""
		Gets the ring atoms from the particular aromatic amino acid.

		Parameters:
		===========
		- dataframe: the dataframe containing the atom records.
		- aa: the amino acid of interest, passed in as 3-letter string.

		Returns:
		========
		- dataframe: a filtered dataframe containing just those atoms from the 
					 particular amino acid selected. e.g. equivalent to 
					 selecting just the ring atoms from a particular amino 
					 acid. 
		"""

		AA_RING_ATOMS = dict()
		AA_RING_ATOMS['PHE'] = ['CG', 'CD', 'CE', 'CZ']
		AA_RING_ATOMS['TRP'] = ['CD', 'CE', 'CH', 'CZ']
		AA_RING_ATOMS['HIS'] = ['CG', 'CD', 'CE', 'ND', 'NE']
		AA_RING_ATOMS['TYR'] = ['CG', 'CD', 'CE', 'CZ']

		ring_atom_df = self.filter_dataframe(dataframe, 'resi_name', [aa], True)
		ring_atom_df = self.filter_dataframe(ring_atom_df, 'atom', AA_RING_ATOMS[aa], True)

		return ring_atom_df

	def get_ring_centroids(self, ring_atom_df, aa):
		"""
		Computes the ring centroids for each a particular amino acid's ring 
		atoms.

		Ring centroids are computed by taking the mean of the x, y, and z 
		coordinates.

		Parameters:
		===========
		- ring_atom_df: a dataframe computed using get_ring_atoms.
		- aa: the amino acid under study

		Returns:
		========
		- centroid_df: a dataframe containing just the centroid coordinates of 
					   the ring atoms of each residue.
		"""
		centroid_df = ring_atom_df.groupby('resi_num').mean()[['x','y','z']].reset_index()
		centroid_df['resi_name'] = aa

		return centroid_df

	# Continue interaction functions
	def get_aromatic_sulphur_interactions(self):
		RESIDUES = ['MET', 'CYS', 'PHE', 'TYR', 'TRP']
		SULPHUR_RESIS = ['MET', 'CYS']
		AROMATIC_RESIS = ['PHE', 'TYR', 'TRP']

		aromatic_sulphur_df = self.filter_dataframe(self.rgroup_df, 'resi_name', RESIDUES, True)
		distmat = self.compute_distmat(aromatic_sulphur_df)
		interacting_atoms = self.get_interacting_atoms(5.3, distmat)
		interacting_atoms = zip(interacting_atoms[0], interacting_atoms[1])
		interacting_resis = set()
		for (a1, a2) in interacting_atoms:
			resi1 = aromatic_sulphur_df.ix[a1]['resi_name']
			resi2 = aromatic_sulphur_df.ix[a2]['resi_name']

			resi1_num = aromatic_sulphur_df.ix[a1]['resi_num']
			resi2_num = aromatic_sulphur_df.ix[a2]['resi_num']

			if ((resi1 in SULPHUR_RESIS and resi2 in AROMATIC_RESIS) or \
				(resi1 in AROMATIC_RESIS and resi2 in SULPHUR_RESIS)) and \
				resi1_num != resi2_num and \
				abs(resi2_num - resi1_num) >= 2 and \
				(resi2_num, resi1_num) not in interacting_resis:
				interacting_resis.add((resi1_num, resi2_num))

		return interacting_resis

	def get_cation_pi_interactions(self):
		RESIDUES = ['LYS', 'ARG', 'PHE', 'TYR', 'TRP']
		CATION_RESIS = ['LYS', 'ARG']
		PI_RESIS = ['PHE', 'TYR', 'TRP']

		cation_pi_df = self.filter_dataframe(self.rgroup_df, 'resi_name', RESIDUES, True)
		distmat = self.compute_distmat(cation_pi_df)
		interacting_atoms = self.get_interacting_atoms(6, distmat)
		interacting_atoms = zip(interacting_atoms[0], interacting_atoms[1])
		interacting_resis = set()

		for (a1, a2) in interacting_atoms:
			resi1 = cation_pi_df.ix[a1]['resi_name']
			resi2 = cation_pi_df.ix[a2]['resi_name']

			resi1_num = cation_pi_df.ix[a1]['resi_num']
			resi2_num = cation_pi_df.ix[a2]['resi_num']

			if ((resi1 in CATION_RESIS and resi2 in PI_RESIS) or \
				(resi1 in PI_RESIS and resi2 in CATION_RESIS)) and \
				resi1_num != resi2_num and \
				abs(resi1_num - resi2_num) >= 2 and \
				(resi2_num, resi1_num) not in interacting_resis:
				interacting_resis.add((resi1_num, resi2_num))

		return interacting_resis

	##### FUNCTION TO COMPUTE GRAPH ##### 






