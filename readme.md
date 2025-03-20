
# Project -2 Submission for ECE 276A - Manoj Kumar Reddy Manchala

* ### **requirements.txt file is included in the zip folder**


## **How to run:**
* open the **load_data.py**_ file, where you can choose the dataset number
* Edit the dataset number (Don't change the type, it should be string)
* Executing the file **Scan_Mapping.py** gives the scan mapping trajectories
* Executing the file **Occupancy_grid.py** creats an occupancy grid and saves it in a file named **'occupancy_grid_{dataset}.png'**
* Executing the file **Texture_mapping.py** creats a texture map and saves it in a file named **'texture_map_{dataset}.png'**
* Executing the file **optimization.py** optimizes the trajectory, creates the new occupancy grid and texturte maps saves it in a files named **'occupancy_grid_optimized_{dataset}.png'**

## **Files:**
* **load_data** file has the code to load the data (whether it has vicon/cam data or not)
* **Occupancy_grid** file - has the code to create the occupancy grid map
* **Texture_mapping** file - has the code to perform and save texture map
* **optimization** file - has the code to perform fixed point based factor graph optimization - This file has the code for proximity based factor graph optimization if required
* **functions** file - has all the function definitions used in the above files

