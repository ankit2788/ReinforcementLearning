# ReinforcementLearning
Reinforcement Learning Tutorial involving use cases from Finance Domain.

Use cases implemented
- Portfolio Management 
- Order Execution

## The below ReadMe document is structured as below:
- Data folder and contents
- RL Library structure and key contents/ algos used
- How to run
- Sample notebook(s)

### Data Folder and contents
  - Folder Structure as below:
    ![image](https://github.com/ankit2788/ReinforcementLearning/assets/48673475/c07df68e-209b-4153-8a5a-41a23fdcafa5)

  - Contains data for both the use cases (as listed above)

### RL library structure 
RLLibrary is structured as below:
1. **Agents folder** : Contains RL Agents and their definitions. Currently available: Policy gradient based RL Agents
2. **utils** : Common utilities required throughout
3. **FinUseCases**:
  - Individual respective folder for implemented usecases.
  - Each use case will have its own Environment Manager, State defition, Action definition etc.

4. **service**:
  - Python scripts to run the training for individual use case



### HOW TO RUN and SAMPLE NOTEBOOK
1. use environment file: PORTO.yaml to download the required/ dependent libraries. Refer [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
2. Python scripts to train the model are saved under RLLibrary -> services. 
3. Sample notebook for Order Execution can be found as OrderExecV1.0.ipynb
  
  


