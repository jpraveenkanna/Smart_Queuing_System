# Smart Queuing System
The Smart Queuing System analyse number of people in the queue and will direct people when the queue is full. It is tested for three different scenarios (Retail, Manufacturing and Transportation) and to meet the customer constraints it is tested seperately on CPU, Integrated GPU, VPU and FPGA and proposed best hardware choice.

# Sample screenshots from output video
<table>
  <tr align="center">
    <td align="center"><b>Retail Scenario</b></td>
  </tr>
  <tr>
   <td align="center"><img src="/Images/retail_output.png" height=400 width=690></td>
   </tr>
   
 </table>
  

# Testing the hardware
<code>queue_job.sh</code> script is used to submit job to Intel Devcloud and the results are collected once the job is finished.

The workspace is provided with <a href="https://software.intel.com/content/www/us/en/develop/topics/iot/hardware/iei-tank-dev-kit-core.html">IEI Tank AIOT Developer Kit</a>

The list of all available hardwares in the developer kit are mentioned <a href="https://devcloud.intel.com/edge/get_started/devcloud/">here</a>


The list of tested hardwares are as below
<ol>
  <li>CPU - <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel® Core™ i5-6500TE Processor </a></li>
  <li>Integrated GPU - <a href="https://ark.intel.com/products/88186/Intel-Core-i5-6500TE-Processor-6M-Cache-up-to-3-30-GHz-">Intel® HD Graphics 530 </a></li>
  <li>VPU - <a href="https://software.intel.com/en-us/neural-compute-stick">Intel® Neural Compute Stick 2</a></li>
  <li>FPGA - <a href="https://www.ieiworld.com/mustang-f100/en/">IEI Mustang-F100-A10</a></li>
</ol>

# Test results
<table>
  <tr align="center">
    <td align="center"><b>Retail Scenario</b></td>
  </tr>
  <tr>
   <td align="center"><img src="/Images/Retail_performance.png" height=400 width=690></td>
   </tr>
  
  <tr align="center">
  <td align="center"><b>Manufacturing Scenario</b></td>
  </tr>
  <tr>
   <td align="center"><img src="/Images/Manufacturing_performance_graph.png" height=400 width=690></td>
   </tr>
   
   <tr align="center">
   <td align="center"><b>Transportation Scenario</b></td>
  </tr>
  <tr>
   <td align="center"><img src="/Images/transportation_performance_graph.png" height=400 width=690></td>
   </tr>
   
 </table>

# Folder structure
<code>person_detect.py</code> - The main python script which load the model, run inference and process the results.

<code>queue_job.sh</code> - The bash script is used to submit job to Intel Devcloud.

<code>Manufacturing Scenario.ipynb,Retail Scenario.ipynb, Transportation Scenario.ipynb</code> - Provides input parameters for <code>queue_job.sh</code>

<code>results</code> - This folder contains the output of the job containing output video and statistics informations.

<code>Requirements - Overview Docs</code> - Contains client requirement documents

<code>Proposal Document - hardwares selected.pdf</code> - Proposal Document for the hardwares selected

<code>original_videos</code> - Input video for testing.

# Sample output Video

<table>
 <tr>
   <td align="center"><img src="/Images/output_retail.gif"></td>
        </tr></table>

