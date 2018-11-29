# Syncs databases, processes data and generates samples

Full-sized documents with and without handwriting were labeled using Labelbox. The generated 
masks and jsons files are retrieved and made readily accessible in an HDF with **handle_Labelbox.py**
(As we did not pay for Labelbox, we did not have access to their superb API.)

The [NHM PRImA](https://www.primaresearch.org/datasets/NHM) provides pre-labeled in XML format. **read_PRImA.py** reads in the XML data to generate masks for the images and a compiled HDF similar to that made for our Labelbox data in handle_Labelbox.py.

From an HDF dataframe, random samples of the documents are generated and labeled from the collected documents 
and their masks; this is achieved with **create_dataset.py**. The resulting data is pickled and 
used to train the CNN.


### Installation
 * python = 3.6.6
 * opencv3 >= 3.1.0
 * matplotlib >= 3.0.1
 * numpy >= 1.15.4
 * pandas >= 0.23.4
 * seaborn >= 0.9.0
 * sklearn >= 0.20.0
 * urllib3 >= 1.24.1


### Data storage
The original images and their masks are collected by _dated folders_ (of form **DD-MM**) in the **original_data** directory. Each sub-directory has the structure detailed in the table below. Masks are black and white PNGs created by Labelbox to indicate where an element of that type was indicated (white contours).

Directory    | Description
------------ | ----------------------------------------------------------- |
img          | has original jpgs that are uploaded to Labelbox             |
text_mask    | contains mask PNGs associated with handwritten text         |
mark_mask    | holds masks associated with marks (non-textual handwriting) |
machine_mask | masks associated with machine-generated signatures          |


Compiled databases of the original images and masks are saved in **labeled_databases** with the date of last labeled data it contains as the filename (DD-MM.\*). The table below lists the type of files stored in this directory and their purpose. The HDFs are derived from the JSON files of the same name and are produced by the **handle_Labelbox.py** script.

File extension | Description
---------------| ------------------------------------------------------------------------------------- |
\*.json        |files directly downloaded from Labelbox and renamed                                    |
\*.hdf         |each labeled element of a page is given in a row in the dataframe                      |
\*hasHW.hdf    |each unique document is given in a row of the dataframe; specific sampling from \*.hdf |


### Processing of data into HDFs

 1. **_Data directory_**
    * To avoid accidentally re-adding data to Labelbox, new data should be first acculumated in a sub-folder img/ within a folder named with the current date (DD-MM); for example:
    23-10/img/
    Once the new set of data is ready, it should be uploaded to the Labelbox account with the dataset name being that of the date of the folder (DD-MM). Preferably, all images being uploaded should have a unique number following the form pagexxxx.jpg where x represents sequential digits that are larger than the last uploaded file.
 2. **_Labelbox_**
    * Label all data and export as a JSON file with the Generate Masks options on so that the urls associated with the masks are embedded inside the JSON file. (First, Labelbox prepares the file; then, you must click the arrow to begin downloading it.)
    * Copy the long-named file to the labels/ directory and name it DD-MM.json
 3. **_Git repository_** handle_Labelbox.py
     * Use handle_Labelbox.process_labelbox() to create a DD-MM.hdf which will be the basis of your analysis and subsequent dataframes. 

        TO NOTE If data was missing from the JSON files, errors may occur. Known errors will result in a clear message with an error.txt file from which you should visit the listed URLs, resolve the issues, re-export the data, and startover in step 3.

     * After you have successfully, completed creating an HDF, you should sync the masks associated with your data. Masks are saved in text_mask and mark_mask in the MM-DD directory.

          TO NOTE: Read the output message as it may ask you to re-run retrieve_masks(), especially if you added lots of data. This is because of the Labelbox API interface, which restricts the number of pings within some time.

     * After this, feel free to use the convenience functions to generate whatever sub-selection(s) of the master dataframe (DD-MM.hdf) you need for your analysis.


### Creating data for training the neural network
**create_dataset.py** is a convenience function which calls in sequence the following scripts/functions to generate data:
  * defaults.get_default_parser() --- contains the default values (i.e. size fo box, where to save, etc.) used throughout the following scripts/functions. It has a debug flag which will generate plots with the image with the visualization of the criteria used to identify whether a sample has handwriting or not.
  * sampler.random_sampler() -- used to draw random samples of size 150 x 150 default. Within this function, **has_handwriting.py** is called upon to label the
    sample (this is where the debugging plots are).
  * data_mixer() --- from the random samples draws the desired proportion of samples containing handwriting to those containing no handwriting.
  * trainTest_set() --- splits the files from data_mixer() into training and tests set. These are saved in the class specified in training_data.py and are loadable using load.py.

sampler.random_sampler(), data_mixer(), and trainTest_set() are usable on their own with their argument _args_ coming from defaults.get_default_parser(). In 
the debugging mode, only the sampler.random_sampler() needs to be used to generate samples.


### Further information
 Checkout the about_json.txt and about_hdf.txt to learn more about the data (column names, etc.) currently stored and how to access additional features.


### Caveats
 Machine signatures (hwType="mach_sig" in the HDF) may be on pages with or without handwritten elements. As these items, themselves are not handwritten, they have hasHW = 0  in the HDF. When condensing dataframes for page identification, ensure that hasHW's aggregate uses max.
