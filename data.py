from scipy.io import loadmat as loadmat
import os
import subprocess
import pandas as pd
import numpy as np


class RAW():
    """Creates a pandas dataframe out fo eyetrace data
    This DF contains feature1, feature2, xmotion, ymotion, and timesecs
    """
    def __init__(self, data_dir, dt_chopout=0):
        files = os.listdir(data_dir)
        file_paths = [data_dir + '/' + f for f in files]
        raw = [loadmat(f) for f in file_paths]
        [raw[idx].update({'feature0':raw[idx]['feature0'].T})  for idx in range(len(raw))]
        [raw[idx].update({'feature1':raw[idx]['feature1'].T})  for idx in range(len(raw))]
        [raw[idx].update({'ID':files[idx][0:5]})  for idx in range(len(raw))]
        [raw[idx].update({'eye': files[idx][5]})  for idx in range(len(raw))]
        [raw[idx].update({'trace': files[idx][7:10]})  for idx in range(len(raw))]
        [raw[idx].update({'file_ID': files[idx].split('_')[-1].split('.')[0]})  for idx in range(len(raw))]
        [raw[idx].update({'sample_rate': files[idx].split('_')[-3]})  for idx in range(len(raw))]
        self.bad_indexes(raw, dt_chopout)
        self.raw = pd.DataFrame(raw)

    def bad_indexes(self, raw, dt_chopout):
    	"""This is not used now. Used to chop up the data
    	"""
    	N=10
    	if dt_chopout>0:
            for trial_idx in range(len(raw)):
                timesecs = raw[trial_idx]['timesecs'][0]
                feature1 = raw[trial_idx]['feature1'][0]
                timesecs_subs = np.split(timesecs,timesecs.searchsorted(feature1), axis=0)
                for idx in range(len(timesecs_subs)):
                    if len(timesecs_subs) > 2*N:
                        N_ = N
                        timesecs_subs[idx] = timesecs_subs[idx][N_:-N_]
                    else:
                        N_ = int(len(timesecs_subs)/2)
                        timesecs_subs[idx] = timesecs_subs[idx][N_:-N_]

class FILE(dict):
    """Create a container for a file's information.

    FILE object has gettable fields: 
            eol token, usually '\n'
            delimiter, usually ',' or ';'
            header (str, file header), 
            header_lines (int, the number of header lines, usually 1),
            lc (int, the line count of the data file excluding header),
            name (str, the file name, not full path),
            fo (the file object itself from open()),
            type (str, the file type, i.e. txt, csv, etc)
    
    FILE object inherits dict() where each key-val contains a SORT object that
    contains detailed information for each data column

    """

    def __init__(self, filename, delimiter = ',', eol= '\n', header_lines=1):
        """Provide filename, and number of lines in header (default 1)."""
        def linecount(self, filename):
            """Get linecount."""
            wc_str = ["wc", "-l", filename]
            if (0 == subprocess.check_call(wc_str)):
                wc = subprocess.check_output(["wc", "-l", filename])
                print('\n')
                numlines = int(wc.decode().split(' ')[0])
            else:   # this is slow for big files
                numlines = len(self.__fo.readlines())
                self.__fo.seek(0)
                print("wc -l failed. Counting lines using readline() instead.")
            return numlines
        super().__init__()
        self.__header_lines = header_lines
        self.__fo = open(filename)
        self.__lc = linecount(self, filename) - header_lines
        self.__name = (filename.split('/'))[-1]
        self.__header = self.__fo.readline()
        self.__delimiter = delimiter
        self.__eol = eol
        # type is useful for building dicts of data from associated files
        self.__member = filename.split('/')[-1].split('.')[-2]
        self.__type = filename.split('/')[-1].split('.')[-1]

    @property
    def delimiter(self):
        """Get the file header as one string."""
        return self.__delimiter
        
    @property
    def eol(self):
        """Get the file header as one string."""
        return self.__eol

    @property
    def header_lines(self):
        """Get the file header as one string."""
        return self.__header_lines

    @property
    def header(self):
        """Get the file header as one string."""
        return self.__header

    @property
    def name(self):
        """Get the name of the file."""
        return self.__name

    @property
    def lc(self):
        """Get line count."""
        return self.__lc

    @property
    def data(self):
        """Get enumerated file data beginning after header (line 1, not 0)."""
        return self.__data

    @property
    def file_member(self):
        """Get file type."""
        return self.__file_member
    
    @property
    def file_type(self):
        """Get file type."""
        return self.__file_type

    @property
    def fo(self):
        """Get opened file object."""
        return self.__fo

    def generate_db_of_file(self):
        """Fills out the dict() for a structured data file with unequal 
        number ofrows for each column. Data of each column is of type
        either string or float."""

        
        columns = self.header.strip(self.eol).split(self.delimiter)
        for i in range(len(columns)):
            self.update({columns[i]: []})
        print(columns)
        
        for i in range(self.lc):
            line = self.fo.readline()
            linelist = line.strip(self.eol).split(self.delimiter)
            
            for j in range(len(columns)):
                try:
                    if not ((linelist[j] == '.') or (linelist[j] == '')):
                        self[columns[j]].append(float(linelist[j]))
                except:
                    bad_names = len(columns) - len(linelist)
                    # This flags a cleanup issue known for patients_stats.csv
                    es1 = 'Number of columns of patient_stats.csv does not match\n'
                    es2 = 'number of data fields. There are {0} column names that likely'
                    es3 = 'have illegal commas. Please correct patient_stats.csv.'
                    es = es1 +es2 + es3
                    print(es.format(bad_names))
                    self.fo.close()
                    return
        self.fo.close()
