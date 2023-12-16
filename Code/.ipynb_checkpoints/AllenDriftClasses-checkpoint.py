import os
import numpy as np
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

from sys import platform
DRIVE_PATH = ''
if platform == "darwin":
    DRIVE_PATH = os.path.join(os.path.dirname(os.getcwd()), 'Data')
elif platform == "win32":
    DRIVE_PATH = 'Y:\Allen\Data_Arshia'


MANIFEST_FILE = os.path.join(DRIVE_PATH, 'BrainObservatoryCache', 'manifest.json')
BOC = BrainObservatoryCache(manifest_file=MANIFEST_FILE)
    
    
class Analyzer:
    
    def data_downloader(area, stimulus = 'natural_movie_one', path = DRIVE_PATH):
        """ This code gets all the mice with specific experiments area and saves their data"""
        import pandas as pd
        import os

        # Properties of the experiments 
        exp_container = pd.DataFrame(BOC.get_ophys_experiments(stimuli=[stimulus], targeted_structures=[area]))

        # Related experiment container ids
        ids = list(set(exp_container['experiment_container_id'].tolist()))


        # replacing all the int values in dataframe with string type to use them as filenames while saving in computer 
        exp_container['imaging_depth'] = exp_container['imaging_depth'].astype(str)
        exp_container['experiment_container_id'] = exp_container['experiment_container_id'].astype(str)


        # Making names for the mice instances files
        file_names = exp_container.apply(lambda row: '_'.join(row[['experiment_container_id', 'targeted_structure','imaging_depth']]), axis=1).tolist()


        # Directory to store data 
        parent_dir = path
        
        if not os.path.exists(parent_dir):
            os.mkdir(parent_dir)
            
        try:
            data_directory = os.path.join(parent_dir, area)
            os.mkdir(data_directory)
            print("Data Directory created successfully.")
        except FileExistsError:
            print("Data Directory already exists.")
        

        import numpy as np
        # For loop to create instances and saving them with their specific name
        for index in range(len(ids)):

            file_dir = os.path.join(data_directory, file_names[index] + ".npz")
            
            
            # Save the instance to a file   
            if not os.path.exists(file_dir):
                mouse_obj = Mouse(ec_id=ids[index])
                
                # with open(file_dir, 'w') as json_file:
                    # json.dump(mouse_obj.__dict__, json_file)
                    
                np.savez(file_dir, **mouse_obj.__dict__)
            print(str(index+1) + " out of " + str(len(ids)) + " ," + str((index+1)/len(ids)) + '%', end='\r')
            
    
    def get_mice(stimulus='natural_movie_one', areas = ['VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'], Nmice = None):
    
        file = [] 
        
        imaging_depths = [depth for depth in BOC.get_all_imaging_depths() if  100 <= depth <= 300]

        data_frame = pd.DataFrame(BOC.get_ophys_experiments(stimuli=[stimulus], imaging_depths=imaging_depths, targeted_structures=areas))
        
        donor_name_list = data_frame['specimen_name'].tolist()
        # print(donor_name_list, '\n')
        # print(len(donor_name_list), '\n')
        
        indices = [index for index, name in enumerate(donor_name_list) if name.split(';')[1] == 'Camk2a-tTA']

        ec_ids = list(set(data_frame.iloc[indices]['experiment_container_id'].tolist()))
        # print(str(len(ec_ids)) + '\n')
        # print(len(set(ec_ids)))
        
        # creating the mouse instances
        mouses = []
        if Nmice == None:
            Nmice = len(ec_ids)
            
        for index, ec_id in enumerate(ec_ids):
            
            print(str(int(100* index/Nmice)) + ' %', end='\r')
            mouses.append(Mouse(ec_id))
            
            if index + 1 == Nmice :
                break
            
        
        return mouses
    
    
class Mouse:
    """
    A class representing a mouse with access to the Allen Brain Observatory data.
    """
    
    def __init__(self, ec_id):
        """
        Initialize a Mouse instance with the given experiment container id (ec_id).
        """ 
        
        self.experiment_container = BOC.get_experiment_containers(ids=[ec_id])[0]
        
        file_name = str(ec_id) + '_' + self.experiment_container['targeted_structure'] + '_' + str(self.experiment_container['imaging_depth']) + '.npz'
        file_expected_dir = os.path.join(DRIVE_PATH, self.experiment_container['targeted_structure'], file_name)
        
        # checking if the data related to this id exist . Then loading 
        if os.path.exists(file_expected_dir):
            mouse_dict = np.load(file_expected_dir, allow_pickle=True)
            
            mouse_dict = {key : value for key,value in mouse_dict.items()}

            for key, value in mouse_dict.items():
                if len(value.shape) == 0:
                    mouse_dict[key] = value.item()
                elif len(value.shape) == 1:
                    mouse_dict[key] = list(value)
                    
            self.__dict__ = mouse_dict
            self.ec_id = ec_id
        
        # getting the data related to the ec_id 
        else:
            self.ec_id = ec_id
            self.get_data_set()
            self.get_dff()
            # del self.data_sets
    

    
    def __str__(self):
        self.name = "Mouse_" + str(self.experiment_container['id']) + "_" + self.experiment_container['targeted_structure'] + "_" + str(self.experiment_container['imaging_depth'])
        return self.name
  
    def get_data_set(self):
        ec_id = self.ec_id
        """
        Retrieve and process the data set for the given experiment container ID.
        """
       
        experiments_dict = BOC.get_ophys_experiments(experiment_container_ids=[ec_id])
        keys_to_select = ['id', 'acquisition_age_days', 'session_type']

        self.experiments_dict = [
            {key: experiment[key] for key in keys_to_select} 
            for experiment in experiments_dict
        ]
        
        for experiment in self.experiments_dict:
            experiment['experiment_stimuli'] = BOC.get_ophys_experiment_stimuli(experiment['id'])

        self.experiments_dict.sort(key=lambda x: x['acquisition_age_days'])
        
        self.data_sets = [
            
            BOC.get_ophys_experiment_data(ophys_experiment_id=experiment['id'])
            for experiment in self.experiments_dict
        ]
        
        return self.data_sets
   
    def mutal_cell_ids(self):
        data_sets = self.get_data_set()
        
        cell_ids = set(data_sets[0].get_cell_specimen_ids())

        for data_set in data_sets:
            cell_ids = cell_ids.intersection(data_set.get_cell_specimen_ids())
        
        
        return list(cell_ids)
    
    def get_dff(self, stimulus = 'natural_movie_one'):
            DFF = np.zeros([len(self.mutal_cell_ids()), len(self.data_sets), 10, 900])

            for index, data_set in enumerate(self.data_sets):
                ts, dff = data_set.get_dff_traces()
                dff = dff[data_set.get_cell_specimen_indices(self.mutal_cell_ids()), :]

                stimulus_table = data_set.get_stimulus_table(stimulus)
                table = []
                for i in range(10):
                    repeat_i = stimulus_table[stimulus_table['repeat'] == i]
                    table.append( {'repeat' : i + 1, 'start' : repeat_i.start.to_list()[0] , 'end' : repeat_i.end.to_list()[-1] } )
                stimulus_table = table

                temp = np.zeros([dff.shape[0], 10, 900])

                for i in range(10):
                    start = table[i]['start']
                    end = start + 900

                    temp[:, i, :] = dff[:, start:end]

                dff = temp

                DFF[:, index, :, :] = dff

            self.dff = DFF







