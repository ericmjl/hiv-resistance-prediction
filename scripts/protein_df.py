class ProteinDataFrame(object):
    """docstring for ProteinDataFrame"""
    def __init__(self, handle):
        super(ProteinDataFrame, self).__init__()
        self.handle = handle
        self.protein_df = self.parse_pdb()
        self.alpha_carbons = self.extract_alpha_carbons()

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
        return atomic_d

    def extract_alpha_carbons(self):
        c_alpha = self.protein_df[self.protein_df['atom'] == 'CA']
        c_alpha.reset_index(drop=True, inplace=True)
        c_alpha = c_alpha[c_alpha['chain_id'] == 'A']

        self.alpha_carbons = c_alpha

    def mutate_position(self, position, new_resi):
        """
        Parameters:
        ===========
        - position: (int) the amino acid position. Begins at 1, not 0.
        - new_resi: (str) the 3-letter amino acid to mutate to.
        """

        assert isinstance(new_resi, str), 'new_resi must be a string'
        assert isinstance(position, int), 'position must be an integer'
        assert len(new_resi) == 3, 'new_resi must be a 3-letter string'

        mutated = c_alpha.copy()
        mutated.ix[position - 1, 'resi_name'] = new_resi

        return mutated


