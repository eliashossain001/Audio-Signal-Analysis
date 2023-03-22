#!/usr/bin/env python
# coding: utf-8

# # Import necessary libraries

# In[1]:


import os
import pandas as pd
import librosa
import IPython.display as ipd
import time
import matplotlib.pyplot as plt


# In[31]:


# pip install --upgrade numba


# # Analysis 1: Audio with duration

# In[2]:


# Set directory path and extension of audio files
directory_path = "/home/elias/clean_data-female/wav"
extension = ".wav"
min_duration = 1.0
# Create a list of file paths
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(extension)]

# Initialize empty lists for file metadata
filenames = []
durations = []

# Loop through the files and extract metadata
for file_path in file_paths:
    filename = os.path.basename(file_path)
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
    except:
        # If there is an error loading the file, set duration to -1
        duration = -1
    filenames.append(filename)
    durations.append(duration)

# Create a Pandas DataFrame with the metadata
df_aud_dur = pd.DataFrame({"filename": filenames, "duration": durations})

# Filter the DataFrame based on your criteria
broken_files = df_aud_dur[df_aud_dur["duration"] == -1]
distorted_files = df_aud_dur[(df_aud_dur["duration"] > 0) & (df_aud_dur["duration"] < min_duration)]
short_files = df_aud_dur[df_aud_dur["duration"] < min_duration]

df_aud_dur.head()


# The above columns show the audio and their duration

# ## Analysis 2: Audio less than 1 second

# In[33]:


# Print the filenames of short files
for filename in short_files["filename"]:
    print(f"{filename} is less than {min_duration} seconds.")


# # Analysis 3: Listen the particular audio

# In[34]:


wavs_list=['/home/elias/clean_data-female/wav/034ce35d-c04e-4866-90cf-4106f6bed7c6.wav', 
      '/home/elias/clean_data-female/wav/8512f9ea-2822-46e8-876d-d0d560c19115.wav']

for wav in wavs_list:
    ipd.display(ipd.Audio(wav, autoplay=True))
    time.sleep(5) # next autoplay starts in 5s


# In[35]:


print(df_aud_dur[df_aud_dur["filename"] == "8512f9ea-2822-46e8-876d-d0d560c19115.wav"]["duration"])


# In[36]:


# !pip install librosa==0.7.2


# # Analysis 4: Waveform and spectogram analysis 

# In[37]:


import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load audio file
y, sr = librosa.load('/home/elias/clean_data-female/wav/8512f9ea-2822-46e8-876d-d0d560c19115.wav', sr=48000)

# Plot waveform
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title('Waveform')

# Plot spectrogram
spec = librosa.stft(y)
spec_db = librosa.amplitude_to_db(abs(spec))
plt.figure(figsize=(10, 4))
librosa.display.specshow(spec_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram')


# The above waveform and spectogram are based on the audio less than 1 second. 

# # Analysis 5: Frame size wise silence detection

# In[3]:


# Set directory path and extension of audio files
directory_path = "/home/elias/clean_data-female/wav"
extension = ".wav"
min_duration = 1.0

# Create a list of file paths
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(extension)]

# Initialize empty lists for file metadata
filenames = []
durations = []
silence_begin = []
silence_end = []

# Loop through the files and extract metadata
for file_path in file_paths:
    filename = os.path.basename(file_path)
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # detect the non-silent intervals of the audio signal
        intervals = librosa.effects.split(y)
        # calculate the duration of the silence at the beginning and end of the audio
        if len(intervals) > 0:
            start_silence = intervals[0][0] / sr
            end_silence = (len(y) - intervals[-1][-1]) / sr
        else:
            start_silence = 0
            end_silence = 0
    except:
        # If there is an error loading the file, set duration and silence to -1
        duration = -1
        start_silence = -1
        end_silence = -1
        
    filenames.append(filename)
    durations.append(duration)
    silence_begin.append(start_silence)
    silence_end.append(end_silence)

# Create a Pandas DataFrame with the metadata
aud_frame_df = pd.DataFrame({"filename": filenames, "duration": durations, "silence_begin": silence_begin, "silence_end": silence_end})

# Filter the DataFrame based on your criteria
broken_files = aud_frame_df[aud_frame_df["duration"] == -1]
distorted_files = aud_frame_df[(aud_frame_df["duration"] > 0) & (aud_frame_df["duration"] < min_duration)]
short_files = aud_frame_df[aud_frame_df["duration"] < min_duration]
aud_frame_df.head()


# Each audio has a column called 'Silence Being and Silence Ending' for an audio. For file number 0, we can see that the silence is 0.00 but the silence end is found to be 0.43, which indicates that the particular audio was silent for 0.43 milliseconds when the speaking stopped.

# # Analysis 6 (a): Calculate total and actual silence unsorted

# In[4]:


# Set directory path and extension of audio files
directory_path = "/home/elias/clean_data-female/wav"
extension = ".wav"
min_duration = 1.0

# Create a list of file paths
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(extension)]

# Initialize empty lists for file metadata
filenames = []
durations = []
total_silence = []
actual_speech = []

# Loop through the files and extract metadata
for file_path in file_paths:
    filename = os.path.basename(file_path)
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        # detect the non-silent intervals of the audio signal
        intervals = librosa.effects.split(y) 
        # calculate the duration of the silence at the beginning and end of the audio
        if len(intervals) > 0:
            start_silence = intervals[0][0] / sr
            end_silence = (len(y) - intervals[-1][-1]) / sr
        else:
            start_silence = 0
            end_silence = 0
        
        # calculate the total duration of silence in the audio
        total_silence_duration = 0
        for i in range(len(intervals)-1):
            gap_duration = (intervals[i+1][0] - intervals[i][-1]) / sr
            if gap_duration > 0.1: # consider gaps longer than 100ms as silence
                total_silence_duration += gap_duration
        
        # calculate the actual speech duration as a percentage of the total duration
        actual_speech_duration = (duration - total_silence_duration) 
        
        percentage = (actual_speech_duration / duration) * 100
        
    except:
        # If there is an error loading the file, set duration, silence, and actual speech to -1
        duration = -1
        start_silence = -1
        end_silence = -1
        total_silence_duration = -1
        actual_speech_duration = -1
        
    filenames.append(filename)
    durations.append(duration)
    total_silence.append(total_silence_duration)
    actual_speech.append(percentage)

# Create a Pandas DataFrame with the metadata
actual_speech_df = pd.DataFrame({"filename": filenames, "duration": durations, "total_silence": total_silence, "actual_speech": actual_speech})

# Filter the DataFrame based on your criteria
broken_files = actual_speech_df[actual_speech_df["duration"] == -1]
distorted_files = actual_speech_df[(actual_speech_df["duration"] > 0) & (actual_speech_df["duration"] < min_duration)]
short_files = actual_speech_df[actual_speech_df["duration"] < min_duration]

# Print the DataFrame with actual speech duration as a percentage
actual_speech_df["actual_speech"] = actual_speech_df["actual_speech"].round(2)
actual_speech_df.head(20)


# librosa.get_duration: librosa.get_duration(y, sr) works by dividing the length of the audio signal y (in samples) by the sampling rate sr (in Hz). This gives the duration of the audio signal in seconds. For example, if y has length 48000 (which is the number of samples in one second of audio at a sampling rate of 48000 Hz), and sr is 48000, then librosa.get_duration(y, sr) will return 1.0 because the audio signal is one second long. If y has length 96000 and sr is 48000, then librosa.get_duration(y, sr) will return 2.0 because the audio signal is two seconds long.
# 

# In[40]:


sorted(actual_speech_df['actual_speech'])


# # Analysis 6: (b) Calculate total and actual silence sorted

# In[5]:


import os
import librosa
import pandas as pd

# Set directory path and extension of audio files
directory_path = "/home/elias/clean_data-female/wav"
extension = ".wav"
min_duration = 1.0

# Create a list of file paths
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(extension)]

# Initialize empty lists for file metadata
filenames = []
durations = []
total_silence = []
actual_speech = []

# Loop through the files and extract metadata
for file_path in file_paths:
    filename = os.path.basename(file_path)
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        # detect the non-silent intervals of the audio signal
        intervals = librosa.effects.split(y) 
        # calculate the duration of the silence at the beginning and end of the audio
        if len(intervals) > 0:
            start_silence = intervals[0][0] / sr
            end_silence = (len(y) - intervals[-1][-1]) / sr
        else:
            start_silence = 0
            end_silence = 0
        
        # calculate the total duration of silence in the audio
        total_silence_duration = 0
        for i in range(len(intervals)-1):
            gap_duration = (intervals[i+1][0] - intervals[i][-1]) / sr
            if gap_duration > 0.1: # consider gaps longer than 100ms as silence
                total_silence_duration += gap_duration
        
        # calculate the actual speech duration as a percentage of the total duration
        actual_speech_duration = (duration - total_silence_duration) 
        
        percentage = (actual_speech_duration / duration) * 100
        
    except:
        # If there is an error loading the file, set duration, silence, and actual speech to -1
        duration = -1
        start_silence = -1
        end_silence = -1
        total_silence_duration = -1
        actual_speech_duration = -1
        
    filenames.append(filename)
    durations.append(duration)
    total_silence.append(total_silence_duration)
    actual_speech.append(percentage)

# Create a Pandas DataFrame with the metadata
actual_speech_df = pd.DataFrame({"filename": filenames, "duration": durations, "total_silence": total_silence, "actual_speech": actual_speech})

# Sort the DataFrame by actual_speech in descending order
actual_speech_df = actual_speech_df.sort_values(by='actual_speech', ascending=False)

# Filter the DataFrame based on your criteria
broken_files = actual_speech_df[actual_speech_df["duration"] == -1]
distorted_files = actual_speech_df[(actual_speech_df["duration"] > 0) & (actual_speech_df["duration"] < min_duration)]
short_files = actual_speech_df[actual_speech_df["duration"] < min_duration]

# Print the DataFrame with actual speech duration as a percentage
actual_speech_df["actual_speech"] = actual_speech_df["actual_speech"].round(2)
print(actual_speech_df.head(20))


# In[6]:


actual_speech_df.tail(20)


# # Data Distribution

# In[7]:


import matplotlib.pyplot as plt

# Get the actual speech percentage values
actual_speech_percentages = actual_speech_df["actual_speech"]

# Set the bins and range for the histogram plot
bins = [0, 10, 20, 50, 70, 100]
range = (0, 100)

# Create the histogram plot
plt.hist(actual_speech_percentages, bins=bins, range=range)

# Set the plot title and labels
plt.title("Distribution of Actual Speech Percentage")
plt.xlabel("Actual Speech Percentage")
plt.ylabel("Frequency")

# Show the plot
plt.show()


# This histogram plot with the x-axis showing the actual speech percentage intervals and the y-axis showing the frequency of audio files in each interval.

# In[8]:


import seaborn as sns

# Create a cumulative distribution plot using seaborn
sns.ecdfplot(actual_speech_df["actual_speech"], stat="proportion")

# Add labels and title
plt.xlabel("Actual Speech Duration (%)")
plt.ylabel("Proportion of Files")
plt.title("Cumulative Distribution of Actual Speech Duration")

# Show the plot
plt.show()


# This cumulative distribution plot that shows the proportion of files that have an actual speech duration less than or equal to a certain percentage. The x-axis represents the actual speech duration as a percentage, and the y-axis represents the proportion of files. The plot shows a step function that increases as we move from left to right, indicating the cumulative distribution of actual speech duration.
# 
# Note that we are using the stat="proportion" parameter to create a cumulative distribution plot instead of an empirical cumulative distribution plot. This means that the y-axis will show the proportion of files rather than the actual cumulative proportion.

# In[9]:


# create boxplot
sns.boxplot(data=actual_speech_df, x='actual_speech')


# In[10]:


# create violinplot
sns.violinplot(data=actual_speech_df, x='actual_speech')


# In[11]:


# create scatter plot
sns.scatterplot(data=actual_speech_df, x='total_silence', y='actual_speech')


# In[12]:


# select numeric columns
numeric_cols = ['duration', 'total_silence', 'actual_speech']

# create correlation matrix
corr_matrix = actual_speech_df[numeric_cols].corr()

# create heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Analysis 7: Check the potential issues

# In[ ]:





# In[ ]:





# In[49]:


# Set directory path and extension of audio files
directory_path = "/home/elias/clean_data-female/wav"
extension = ".wav"
min_duration = 1.0
min_silence = 0.5

# Create a list of file paths
file_paths = [os.path.join(directory_path, file) for file in os.listdir(directory_path) if file.endswith(extension)]

# Initialize empty lists for file metadata
filenames = []
durations = []
silence_begin = []
silence_end = []
silence_total = []

# Loop through the files and extract metadata
for file_path in file_paths:
    filename = os.path.basename(file_path)
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # detect the non-silent intervals of the audio signal
        intervals = librosa.effects.split(y)
        # calculate the duration of the silence at the beginning and end of the audio
        if len(intervals) > 0:
            start_silence = intervals[0][0] / sr
            end_silence = (len(y) - intervals[-1][-1]) / sr
            total_silence = sum([(i[1]-i[0])/sr for i in intervals])
        else:
            start_silence = 0
            end_silence = 0
            total_silence = 0
    except:
        # If there is an error loading the file, set duration and silence to -1
        duration = -1
        start_silence = -1
        end_silence = -1
        total_silence = -1
        
    filenames.append(filename)
    durations.append(duration)
    silence_begin.append(start_silence)
    silence_end.append(end_silence)
    silence_total.append(total_silence)

# Create a Pandas DataFrame with the metadata
df = pd.DataFrame({"filename": filenames, "duration": durations, "silence_begin": silence_begin, "silence_end": silence_end, "silence_total": silence_total})

# Filter the DataFrame based on your criteria
broken_files = df[df["duration"] == -1]
distorted_files = df[(df["duration"] > 0) & (df["duration"] < min_duration)]
short_files = df[df["duration"] < min_duration]
silent_files = df[df["silence_total"] >= min_silence]

# Check for any potential issues
if len(broken_files) > 0:
    print(f"{len(broken_files)} file(s) could not be loaded.")
if len(distorted_files) > 0:
    print(f"{len(distorted_files)} file(s) have a duration less than {min_duration} seconds.")
if len(short_files) > 0:
    print(f"{len(short_files)} file(s) have a duration less than {min_duration} seconds.")
if len(silent_files) > 0:
    print(f"{len(silent_files)} file(s) have more than {min_silence} seconds of total silence.")


# # Analysis 8: More in-depth analysis

# In[50]:


# Compute statistics for duration and silence
duration_stats = df["duration"].describe()
begin_silence_stats = df["silence_begin"].describe()
end_silence_stats = df["silence_end"].describe()

# Plot histograms of duration and silence
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
df["duration"].hist(ax=axes[0, 0])
df["silence_begin"].hist(ax=axes[0, 1])
df["silence_end"].hist(ax=axes[1, 0])
df["silence_total"].hist(ax=axes[1, 1])
axes[0, 0].set_xlabel("Duration (s)")
axes[0, 1].set_xlabel("Silence at Beginning (s)")
axes[1, 0].set_xlabel("Silence at End (s)")
axes[1, 1].set_xlabel("Total Silence (s)")
axes[0, 0].set_ylabel("Count")
axes[0, 1].set_ylabel("Count")
axes[1, 0].set_ylabel("Count")
axes[1, 1].set_ylabel("Count")
fig.tight_layout()

# Plot boxplots of duration and silence
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
# df.boxplot(column=["duration"], ax=axes[0])
# df[["silence_begin", "silence_end", "silence_total"]].boxplot(ax=axes[1:], showfliers=False)
# axes[0].set_ylabel("Duration (s)")
# axes[1].set_ylabel("Silence (s)")
# axes[2].set_ylabel("Silence (s)")
# fig.tight_layout()

# Identify potential problematic files based on the metadata
problematic_files = df[(df["duration"] < duration_stats["25%"]) | 
                       (df["silence_begin"] > begin_silence_stats["75%"]) | 
                       (df["silence_end"] > end_silence_stats["75%"]) |
                       (df["silence_total"] > df["duration"] * 0.5)]
print("Potential problematic files:")
print(problematic_files)


# I have computed some basic statistics (e.g., mean, median, min, max, quartiles) for the duration and amount of silence at the beginning, end, and total of each audio file. It then plots histograms and boxplots of these metadata to visualize their distribution and outliers.

# # Analysis 9: Finding outliers data point

# To ferret out the outliers lurking within our dataset, we'll hone in on the actual speech percentage. Should the speech percentages fall outside the boundaries we've set, it'll be cast aside as an outlier. Think of it like detecting a black sheep in a flock - we're scanning for the exceptional cases that don't fit the norm.

# In[13]:


actual_speech_df.head(10)


# Set 0.04 for the first quartile, instead of 25%

# In[26]:


Q1 = actual_speech_df['actual_speech'].quantile(0.04)
Q3 = actual_speech_df['actual_speech'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


# In[27]:


outliers = actual_speech_df[(actual_speech_df['actual_speech'] < lower_bound) | (actual_speech_df['actual_speech'] > upper_bound)]
outliers


# In[28]:


len(outliers)


# When working with a dataset, it's important to identify any outliers that might be lurking within it. These outliers can throw off our analysis, leading to skewed results and false conclusions. One way to detect outliers is to focus on the actual speech percentage.
# 
# By setting boundaries for the acceptable range of speech percentages, we can identify any data points that fall outside of that range. These outlying data points, known as outliers, can then be flagged and removed from our analysis, allowing us to more accurately interpret the remaining data.
# 
# To better understand this concept, think of it like scanning a flock of sheep for a black sheep. The black sheep stands out from the rest of the flock, just as outliers stand out from the rest of the data. By pinpointing these exceptional cases that don't fit the norm, we can better understand the trends and patterns in our dataset.
# 
# When analyzing a dataset, it's important to adjust our outlier detection methods based on the size and distribution of the data. For example, let's say we have a dataset with a range of percentages, including some very small values like 0.004. If we set narrow boundaries for outlier detection, we may end up with a large number of outliers - perhaps even 8 in this case.
# 
# However, when we adjust our approach to account for the size and distribution of the data, we may find that the number of outliers decreases dramatically. For instance, if we set a 25% threshold in the first quartile of the data, we may find that there are no outliers at all, since the data is not as large and the threshold is more lenient.
# 
# By adapting our outlier detection methods to fit the specific characteristics of our dataset, we can ensure that our analysis is accurate, relevant, and reliable.

# # Elias Hossain
# ## North South University
# ### Department of Electrical and Computer Engineering
# 
