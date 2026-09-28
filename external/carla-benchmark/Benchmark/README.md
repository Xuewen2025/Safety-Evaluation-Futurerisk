## Directory Structure

*   `reports/`: Contains XML files of the crash reports downloaded from [NHTSA Crash Viewer](https://crashviewer.nhtsa.dot.gov/CrashAPI).

*   `summary/`: Contains text files summarizing the reports with all the relevant information needed to reconstruct the crash incident.

*   `maps/`: Contains the OpenDRIVE map files (`.xodr`) used in the simulations.

*   `simulations/`: Contains JSON files that define the simulation scenarios with vehicles' starting position, direction, speed and the trajectory waypoints.

## Quick Benchmark Demo
If you want to run a quick check on the Benchmark, you need to use the docker image. Follow the instructions in `Reconstruction-Pipeline/README.md`.

After building and running the docker, this will drop you into an interactive `bash` shell inside the container, in the `/home/carla` directory.

Once inside the container's shell, you can perform the demo on the Benchmark. 

To check the benchmark files, move to `/home/carla/PythonAPI/util/Benchmark` within the container.

```bash
# Navigate to the directory containing the scripts
cd /home/carla/PythonAPI/util/Benchmark
```

If you run `ls -F` command, then you can see the following files.
```sh
maps/  README.md  reports/  simulations/  summary/
```
You will need the specific crash_id of the scenario to see the demo of that crash. The `crash_id` can be found from the filenames in the `Benchmark/` subdirectories (e.g., for `simulation_510002.json`, the ID is `510002`).

To run the scenario from the Benchmark, move to `/home/carla/PythonAPI/util/` within the container.

```bash
# Navigate to the directory containing the scripts
cd ..
```

Now run the following command with the specific crash_id you want to see the demo of. 
```sh
./run_benchmark.sh <crash_id>
```
It will take some time to load the map and play the simulation.

## Usage (Benchmark)
To work with these benchmarks, you need to copy this folder into `your_carla_route/PythonAPI/util/` along with all the scripts of the `Reconstruction-Pipeline/scripts`.

After copying, navigate to the `util` directory.

You can change the control logic of one of the vehicles with your custom agent to verify the performance of your agent.
To do that, you will need to:
1. Import your agent's class into `ns_SV_junction.py` and `ns_SV_nonjunction.py`.
2. Initialize your agent's class within these files.
3. Change the control logic of the ego vehicle with your agent's control logic.
4. Generate the simulation scenario for the custom agent.