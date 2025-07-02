import os
from lgmpy import Lgm_Vector
import lgmpy.Lgm_Wrap as lgm_lib
from ctypes import pointer, c_int
import spacepy.datamodel as dm
import numpy as np
import datetime as dt

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

    output_lgm = Lgm_Vector.Lgm_Vector()
    output = np.zeros_like(loaded_data['TestPosition'], dtype=np.float64)
    diff = np.zeros(len(loaded_data['DateTime']))
    nPass = 0
    nFail = 0
    for i in range(len(loaded_data['DateTime'])):
        transflag = c_int(lgm_lib.__dict__[f"{loaded_data['SysIn'][i]}_TO_{loaded_data['SysOut'][i]}"])
        lgm_vec = Lgm_Vector.Lgm_Vector(*loaded_data['TestPosition'][i,:])
        time = loaded_data['DateTime'][i]
        lgm_lib.Lgm_Set_Coord_Transforms(
            int(time.strftime('%Y%m%d')), 
            time.hour + time.minute/60 + time.second/60/60 + time.microsecond/60/60/1000000, c)
        lgm_lib.Lgm_Convert_Coords(pointer(lgm_vec), pointer(output_lgm), transflag, c)
        output[i,:] = [output_lgm.x, output_lgm.y, output_lgm.z]
        diff_vec = output[i,:] - loaded_data['Result'][i,:]
        diff[i] = np.linalg.norm(diff_vec)

        if np.abs(diff[i]) <= 1.0e-5:
            nPass += 1
            print(f"  Test {i} ({diff[i]:.1e}) passed.")
        else:
            nFail += 1
            print(f"  Test {i} ({diff[i]:.1e}) failed.")
