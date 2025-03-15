import numpy as np
from skimage.io import imread
import torch
from torch.utils.data import Dataset
import cv2
import random
import pandas as pd

class DataGenerator(Dataset):
    """Creates Dataset and Picks up a SINGLE Random Sample from Dataset"""

    def __init__(self, device, list_IDs, file_names, input_path, output_path, 
                 dim_X, dim_y, csv_path,
                 shuffle=True, augment_mode=None, random_augment = 0):
        """Initialization
        :param list_IDs: list of all 'label' ids to use in the generator, 0 to Dataset_size-1
        :param file_names: list of file names
        :param input_path: path to input data
        :param output_path: path to PL radio maps
        :param dim_X: input dimensions to resize
        :param dim_y: output dimensions to resize 
        :param csv_path: path to CSV file with center positions
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.list_IDs = list_IDs
        self.file_names = file_names
        self.input_path = input_path
        self.output_path = output_path
        self.dim_X = dim_X
        self.dim_y = dim_y
        self.csv_path = csv_path
        self.batch_size = 1 
        self.shuffle = shuffle
        self.on_epoch_end()
        self.device = device
        self.augment_mode = augment_mode
        self.random_augment = random_augment

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self._generate_X(list_IDs_temp) 
        y = self._generate_y(list_IDs_temp)

        # if self.augment_mode:
        #     X, y = self._apply_augment(X, y)

        if self.random_augment:
            X, y = self._apply_random_augment(X, y)

        elif self.augment_mode:
            X, y = self._apply_augment(X, y)
          
        return torch.from_numpy(X).to(self.device), torch.from_numpy(y).to(self.device)
       
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_IDs_temp):
        filename = self.file_names[list_IDs_temp[0]]
        b, ant, f, sp = self._parse_filename(filename)

        X = imread(self.input_path + filename + ".png")
        
        fspl = self._calculate_fspl(b, ant, f, sp)
        g = self._calculate_ant(b, ant, f, sp)

        X = cv2.resize(X, self.dim_X, interpolation=cv2.INTER_CUBIC)
        fspl = cv2.resize(fspl, self.dim_X, interpolation=cv2.INTER_CUBIC)
        g = cv2.resize(g, self.dim_X, interpolation=cv2.INTER_CUBIC)

        # Stack the FSPL and frequency channels
        X = np.dstack((X, fspl, -g))
        return np.squeeze(X.astype(np.float32))
  
    def _generate_y(self, list_IDs_temp):
        # Load and resize the output map
        y = imread(self.output_path + self.file_names[list_IDs_temp[0]] + ".png")   
        y = cv2.resize(y, self.dim_y, interpolation=cv2.INTER_CUBIC)
        return np.squeeze(y)
    
    def _apply_augment(self, X, y):
        augmentation_options = {
            'original': lambda x: x,
            'rotate_90': lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
            'rotate_180': lambda x: cv2.rotate(x, cv2.ROTATE_180),
            'rotate_270': lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
            'flip_y': lambda x: cv2.flip(x, 1),  
            'flip_x': lambda x: cv2.flip(x, 0), 
            'flip_x_rotate_90': lambda x: cv2.rotate(cv2.flip(x, 0), cv2.ROTATE_90_CLOCKWISE),
            'flip_y_rotate_90': lambda x: cv2.rotate(cv2.flip(x, 1), cv2.ROTATE_90_CLOCKWISE)
        }

        if self.augment_mode in augmentation_options:
            X_1 = augmentation_options[self.augment_mode](X[:,:,:4]) # image + fspl
            X_2 = augmentation_options[self.augment_mode](X[:,:,4]) # -g
            # X = augmentation_options[self.augment_mode](X)
            X = np.dstack((X_1, X_2))
            y = augmentation_options[self.augment_mode](y)
        
        return X, y
    
    def _parse_filename(self, filename):
        # filename 'B5_Ant1_f2_S10'
        b = int(filename.split('_')[0][1:])
        ant = int(filename.split('_')[1][3:])
        f = int(filename.split('_')[2][1:])
        sp = int(filename.split('_')[3][1:])
        return b, ant, f, sp
    
    def _calculate_fspl(self, b, ant, f, sp):
        positions_filename = f"Positions/Positions_B{b}_Ant{ant}_f{f}.csv"
        sampling_positions = pd.read_csv(self.csv_path + positions_filename)

        building_details_filename = f"Building_Details/B{b}_Details.csv"
        building_details = pd.read_csv(self.csv_path+building_details_filename)

        x_ant = sampling_positions["Y"].loc[sp] - 1
        y_ant = sampling_positions["X"].loc[sp] - 1
        W, H = building_details["W"].iloc[0], building_details["H"].iloc[0]

        X_points = np.repeat(np.linspace(0, W - 1, W), H, axis=0).reshape(W, H).transpose()
        Y_points = np.repeat(np.linspace(0, H - 1, H), W, axis=0).reshape(H, W)

        distance_to_transmitter = np.sqrt((x_ant - X_points)**2 + (y_ant - Y_points)**2) * 0.25
        freq_select = [0.868e9, 1.8e9, 3.5e9]

        freq = freq_select[f-1]
        c = 3e8  

        fspl = 20 * np.log10(distance_to_transmitter) + 20 * np.log10(freq) + 20 * np.log10(4 * np.pi / c)
        # print(f"y_ant: {y_ant}, x_ant: {x_ant}, fspl shape: {fspl.shape}")
        # print(f"Postion_B:{b}, ant{ant}, f{f}, sp{sp}")
        if 0 <= y_ant < fspl.shape[0] and 0 <= x_ant < fspl.shape[1]:
            fspl[y_ant, x_ant] = 0
        # print(freq)


        return fspl
    
    def _calculate_ant(self, b, ant, f ,sp):
        positions_filename = f"Positions/Positions_B{b}_Ant{ant}_f{f}.csv"
        sampling_positions = pd.read_csv(self.csv_path + positions_filename)

        building_details_filename = f"Building_Details/B{b}_Details.csv"
        building_details = pd.read_csv(self.csv_path+building_details_filename)

        x_ant = sampling_positions["Y"].loc[sp] - 1
        y_ant = sampling_positions["X"].loc[sp] - 1
        W, H = building_details["W"].iloc[0], building_details["H"].iloc[0]

        X_points = np.repeat(np.linspace(0, W - 1, W), H, axis=0).reshape(W, H).transpose()
        Y_points = np.repeat(np.linspace(0, H - 1, H), W, axis=0).reshape(H, W)

        antenna_pattern_filename = f"Radiation_Patterns/Ant{ant}_Pattern.csv"
        antenna_pattern = np.genfromtxt(self.csv_path + antenna_pattern_filename, delimiter=',', skip_header=1)
        angles = np.mod(-(180 / np.pi) * np.arctan2((y_ant - Y_points), (x_ant - X_points)) + 180 + sampling_positions['Azimuth'].iloc[sp], 360).astype(int)
        angles = np.mod(angles, antenna_pattern.shape[0])

        g = antenna_pattern[angles]
        return g
    
    def _apply_random_augment(self, X, y):
        augmentation_modes = ['original', 'rotate_90', 'rotate_180', 'rotate_270', 
                            'flip_y', 'flip_x', 'flip_x_rotate_90', 'flip_y_rotate_90']
        chosen_mode = random.choice(augmentation_modes)
        
        original_mode = self.augment_mode
        self.augment_mode = chosen_mode

        X_aug, y_aug = self._apply_augment(X, y)
        
        self.augment_mode = original_mode
        
        return X_aug, y_aug