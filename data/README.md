# Syncs databases, processes data and generates samples

Full-sized documents with and without handwriting were labeled using Labelbox. The generated 
masks and jsons files are retrieved and made readily accessible in an HDF with handle_labels.py

The PRImA NHM provides pre-labeled in XML format:
https://www.primaresearch.org/datasets/NHM
read_prima.py reads in the XML data to generate masks for the images.

From the HDF, random samples of the documents are generated and labeled from the collected documents 
and their masks; this is achieved with sample_pages.py. The resulting data is pickled and 
used to train the CNN.

### Installation


 # SYSTEM REQUIREMENTS
 * python 3.x
 * pandas
 * urllib


# ACCESSING DATA
 1. Create a folder (e.g. project) to house the project. 

 2. Create a soft link from your Dropbox folder to a sub-folder known as data:
     ln -s <DROPBOX_PATH>/Training\ Data/ ./data

 3. Within another sub-folder (name not important), clone the github repository:
     https://github.com/dkrft/handwriting_detection

 [In cloned git repository]
 * All data labels are given in the labels/ directory. Files are named based on their most recent checkout date. JSONs come from Labelbox and are processed to make pandas HDFs (with "data" as the key).

 [Dropbox]
 * Data is collected by date (MM-DD) and has this structure:
    - img             has original jpgs that will be uploaded to Labelbox
    - text_mask       contains masks associated with handwritten text
    - mark_mask       holds masks associated with marks (non-textual handwriting)


# PROCESSING NEW DATA

 1. [In Dropbox]
    * To avoid accidentally re-adding data to Labelbox, new data should be first acculumated in a sub-folder img/ within a folder named with the current date (DD-MM); for example:
    23-10/img/

    Once the new set of data is ready, it should be uploaded to the Labelbox account with the dataset name being that of the date of the folder (DD-MM). Preferably, all images being uploaded should have a unique number following the form pagexxxx.jpg where x represents sequential digits that are larger than the last uploaded file.

 2. [Labelbox]
    * Label all data and export as a JSON file with the Generate Masks options on so that the urls associated with the masks are embedded inside the JSON file. (First, Labelbox prepares the file; then, you must click the arrow to begin downloading it.)

 3. [Git repository/handle_labels.py]
     * Copy the long-named file to the labels/ directory and name it DD-MM.json
     * Use create_dataframe() to create a DD-MM.hdf which will be the basis of your analysis and subsequent dataframes. 

        TO NOTE If data was missing from the JSON files, errors may occur. Known errors will result in a clear message with an error.txt file from which you should visit the listed URLs, resolve the issues, re-export the data, and startover in step 3.

     * After you have successfully, completed creating an HDF, you should sync the masks associated with your data. To do this, run retrieve_masks() on the master dataframe (DD-MM.hdf). Masks are saved in text_mask and mark_mask in the MM-DD directory.

          TO NOTE: Read the output message as it may ask you to re-run retrieve_masks(), especially if you added lots of data. This is because of the Labelbox API interface, which restricts the number of pings within some time.

     * After this, feel free to use the convenience functions to generate whatever sub-selection(s) of the master dataframe (DD-MM.hdf) you need for your analysis.


# FURTHER INFORMATION
 Checkout the about_json and about_hdf to learn more about the data currently stored and how to access additional features.

 # CAVEATS
 Machine signatures (hwType=mach_sig) may be on pages with or without handwritten elements. As these items, themselves are not handwritten, they have hasHW = 0. When condensing dataframes for page identification, ensure that hasHW's aggregate uses max.
