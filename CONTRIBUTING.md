# Contributing to COALA

## Preparation

* A Github account that has access to the project.
* A local Python IDE, such as PyCharm and VSCode.
* Link the Github account with the IDE by authentication token.

## Main Workflow 

**Step 1**
* **Create a new issue**: Before starting a new development task, create a corresponding issue in the github project describing the main features and purposes.

* **Pull the project from GitHub**: Each new task should be developed on top of the latest main branch, which can result in less conflicts when the new branch is merged to the main branch as there could be multiple new tasks in parallel.

**Step 2**
* **Update main branch** (make a commit if current branch is not `main`)
  * `git checkout main`
  * `git pull`

* **Create and change to new branch**
  * `git branch new_name`
  * `git checkout new_name`

* **Code Development**
  * add new pythons files
  * modify existing python files

**Step 3**
* **Commit and Push** 
  * adding a meaningful message, e.g., “feat: data integration”
* **Pull Request**
  * create a new PR when new task is finished
  * link to the corresponding issue in the description
  * add a code reviewer\

## Some Tips

* **Code Format:** Using standard pep8 python format\
     * `ctr+alt+L`(Win) or `option+command+L`(Mac) for automatic adjustment in PyCharm
     * or using the autopep8

  


