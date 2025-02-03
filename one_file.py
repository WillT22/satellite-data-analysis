from spacepy import pycdf
import matplotlib.pyplot as plt


#file_path = "C:/Users/wzt0020/Box/Multipoint_Box/REPT Data/April 2017 Storm/rbspa_rel03_ect-rept-sci-l2_20170421_v5.4.0.cdf"
#cdf_data = pycdf.CDF(file_path)
#print(cdf_data)

#print(cdf_data["FESA"].attrs)

ephemeris_path = "C:/Users/Will/Box/Multipoint_Box/REPT Data/April 2017 Storm/ephemeris/rbsp-a_mag-ephem_def-1min-t89d_20170421_v01.cdf"
ephem_data = pycdf.CDF(ephemeris_path)
#print(ephem_data)
Epoch_ephem = ephem_data['Epoch'][:]

Lm_eq = ephem_data['Lm_eq'][:]

# Create the scatter plot
plt.scatter(Epoch_ephem, Lm_eq) 
plt.show()

#for 
for t in range(len(Epoch_ephem)-1):
    #t1 = Epoch_ephem[t]                 #t2 = Epoch_ephem[t+1]
    #Lm_p1 = Lm_eq[t]                    #Lm_p2 = Lm_eq[t+1]
    # Calculate change in time between two points
    delta_t = Epoch_ephem[t+1]-Epoch_ephem[t]
    # Calculate slope of linear equations (convert time delta to float)
    m = (Lm_eq[t+1]-Lm_eq[t])/delta_t.total_seconds()
    
    # Caulcuate time since first point (set first point to t=0)
    time_since_start = Epoch_ephem[t]-Epoch_ephem[0]
    # Calculate y-int of linear equations (convert time delta to float)
    b = Lm_eq[t] - m * time_since_start.total_seconds()