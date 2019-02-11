import streamlit as st
from utils import data as data
from utils import plot_utils as plu

def build_OF_conf(outfile_DB):
	# VA = outfile_DB[' MB_Val_acc'][-1]
	sens = outfile_DB[' MB_sens_at_BVA'][-1]  #TPR recall
	spec = outfile_DB[' MB_spec_at_BVA'][-1]   #TNR selectivity
	T1 = 1 - sens
	T2 = 1 - spec
	array = [[sens, T1],[T2, spec]]
	return plu.confusion(array)

data_dir = "./data/envision_working_traces"
stats_filename = "./data/patient_stats.csv" 
out_filename = "../EnVision/output/SP_1F2.csv"  

R = data.RAW(data_dir, 0.1/480)
stats_file = data.FILE(stats_filename)
stats_file.generate_db_of_file()
expl = plu.EXPLORATION(R.raw)
fig = expl.plot_N(4)
st.pyplot(fig)


## Confusion Matrix Most recent BEST MODEL
outfile = data.FILE(out_filename)
print(type(outfile))
outfile.generate_db_of_file()
plt = build_OF_conf(outfile)
st.pyplot(plt)

# Comparison results Confusion matrix. 
# The following sensitivity and specitivity, type1, and type2 values are hand coded from previous best model
sens = 0.82
spec = 0.68
T1 = 1 - sens
T2 = 1 - spec
array = [[sens, T1],[T2, spec]]
plt2=plu.confusion(array)
st.pyplot(plt2)

 
