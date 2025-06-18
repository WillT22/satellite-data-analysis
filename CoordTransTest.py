import os
from lgmpy import Lgm_CTrans, Lgm_Vector
import lgmpy.Lgm_Wrap as lgm_lib
from ctypes import c_bool, c_int, c_float
import spacepy.datamodel as dm
import numpy as np
import datetime as dt
import re

#%% Start main class
if __name__ == '__main__':
    c = lgm_lib.Lgm_init_ctrans(0) # initializes handle to Lgm_CTrans object (for coordinate transformation)

    lgm_lib.Lgm_Set_CTrans_Options(lgm_lib.LGM_EPH_DE, lgm_lib.LGM_PN_IAU76, c)

    # read test file
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_folder_raw_path = os.path.join(current_script_dir, "../CoordQuickStart/")
    file_folder_path = os.path.normpath(file_folder_raw_path)
    test_file = os.path.join(file_folder_path, "LgmTestTransforms_MMS.txt")

    loaded_data = dm.readJSONheadedASCII(test_file)
    
    datetime_format = "%Y-%m-%dT%H:%M:%S.%f"
    loaded_data['DateTime'] = np.array([dt.datetime.strptime(s, datetime_format) for s in loaded_data['DateTime']], dtype=object)

    Lgm_CTrans_header = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../LANLGeoMag/libLanlGeoMag/Lgm/Lgm_CTrans.h'))
    coord_codes = {}
    pattern = re.compile(r"#define\s+([A-Z0-9_]+_COORDS)\s+(\d+)")
    with open(Lgm_CTrans_header, 'r') as f:
        for line in f:
            match = pattern.match(line.strip())
            if match:
                # Extract the full constant name (e.g., "GSE_COORDS")
                full_name = match.group(1)
                # Extract the integer value (e.g., "7")
                code_str = match.group(2)

                # Remove '_COORDS' suffix and convert to int
                system_name = full_name.replace('_COORDS', '')
                code = int(code_str)
                    
                # Store in the dictionary
                coord_codes[system_name] = code
    
    lgm_vec = Lgm_Vector.Lgm_Vector()
    output_lgm = Lgm_Vector.Lgm_Vector()
    output = np.zeros_like(loaded_data['TestPosition'])
    for i in range(len(loaded_data['SysIn'])):
        transflag = c_int(coord_codes[loaded_data['SysIn'][i]]*100 + coord_codes[loaded_data['SysOut'][i]])
        lgm_vec.x = loaded_data['TestPosition'][i,0]
        lgm_vec.y = loaded_data['TestPosition'][i,1]
        lgm_vec.z = loaded_data['TestPosition'][i,2]
        print(lgm_vec)
        lgm_lib.Lgm_Convert_Coords(lgm_vec, output_lgm, transflag, c)
        print(output_lgm)
        output[i,0] = output_lgm.x
        output[i,1] = output_lgm.y
        output[i,2] = output_lgm.z
    
    