import os
from pyadlml.dataset import fetch_uci_adl_binary

# If Output is existed 
if os.path.exists("./output/OrdonezA.txt"):
  os.remove("./output/OrdonezA.txt")
  
if os.path.exists("./output/OrdonezB.txt"):
  os.remove("./output/OrdonezB.txt")

# Fetch UCI-ADL-BINARY DATA and Analize
data_a = fetch_uci_adl_binary(subject='OrdonezA')
data_b = fetch_uci_adl_binary(subject='OrdonezB')

df_a_devices, df_a_activities = data_a['devices'], data_a['activities']
df_b_devices, df_b_activities = data_b['devices'], data_b['activities']

# Get ready to output results
file_a = open('./output/OrdonezA.txt', 'a')
file_b = open('./output/OrdonezB.txt', 'a')


# Output Activity Count
from pyadlml.stats import activity_count
a_activity_count = activity_count(df_a_activities)
b_activity_count = activity_count(df_b_activities)

file_a.write('Activity Occurrence:\n' + a_activity_count.to_string())
file_b.write('Activity Occurrence:\n' + b_activity_count.to_string())

# Print Activity Count Chart
from pyadlml.plot import plot_activity_count
plot_activity_count(df_a_activities, other=True);
plot_activity_count(df_b_activities, other=True);



# Output Activity Duration
from pyadlml.stats import activity_duration
a_activity_duration = activity_duration(df_a_activities)
b_activity_duration = activity_duration(df_b_activities)

file_a.write('\n\nActivity Duration:\n' + a_activity_duration.to_string())
file_b.write('\n\nActivity Duration:\n' + b_activity_duration.to_string())

# Print Activity Duration Chart
from pyadlml.plot import plot_activity_duration
plot_activity_duration(df_a_activities)
plot_activity_duration(df_b_activities)



# Output Activity Transition
from pyadlml.stats import activity_transition
a_activity_transition = activity_transition(df_a_activities)
b_activity_transition = activity_transition(df_b_activities)

file_a.write('\n\nActivity Transition:\n' + a_activity_transition.to_string())
file_b.write('\n\nActivity Transition:\n' + b_activity_transition.to_string())

# Print Activity Transition Chart
from pyadlml.plot import plot_activity_transitions
plot_activity_transitions(df_a_activities)
plot_activity_transitions(df_b_activities)



# Output Activity Density
from pyadlml.stats import activity_dist
a_activity_dist = activity_dist(df_a_activities, n=1000)
b_activity_dist = activity_dist(df_b_activities, n=1000)

file_a.write('\n\nActivity Density:\n' + a_activity_dist.to_string())
file_b.write('\n\nActivity Density:\n' + b_activity_dist.to_string())

# Print Activity Density Chart   
from pyadlml.plot import plot_activity_density
plot_activity_density(df_a_activities)
plot_activity_density(df_b_activities)

# Close Files
file_a.close()
file_b.close()

print("Success!!!")