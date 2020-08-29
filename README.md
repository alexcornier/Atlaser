# Analysis of electrophysiological and imaging data in the context of sensory information processing in autism

Master's Project, Bio-Computing, University of Bordeaux, 2019-2020

The neocortex plays a central role in processes such as the processing of sensory information, perception or even the control of motor activity. Cortical deficits therefore have dramatic neurological and psychiatric repercussions. The functioning of a cortical circuit results from the combination of the intrinsic properties of the neurons that compose it, the connectivity of these neurons and the properties of these connections. Integrating these three levels of functional complexity, one of the major challenges of contemporary neuroscience, is necessary for understanding the normal and pathological functioning of neural networks and the study of diseases of the central nervous system.

Autism spectrum disorder (ASD) is estimated to affect one in fifty children. Atypical processing of sensory information (for example, tactile, visual and hearing information) is now considered a key phenotype of ASD and can be a key determinant of other basic autistic phenotypes. Information from the different senses are processed in the neocortical circuits and can be measured by electrophysiological or imaging approaches. Measurements of sensory information and perception processing could also provide objective biomarkers essential to complete the evaluation of social, communicational and cognitive/behavioral alterations and to quantify the therapeutic results. Today, re-education is possible depending on the level of severity but there is no targeted therapeutic approach. The object of the project is to improve the functioning of a tool intended to improve the treatment of autism therapeutically.

The objective of this project is to develop software for an in-depth analysis of complex electrophysiological and imaging data. To achieve this, it will be couple existing software with the code produced in the project to extract the various characteristics of this data.

## To start

This code was developed using Python and PyQt5 for the graphical part. Some other Python librairies are needed like PIL, OpenSlide, Numpy, CV2, Scikit-Image, CSV.

A good start is reading first the document **[Rapport___Atlaser.pdf](Rapport___Atlaser.pdf)** provided in this repository.

### Requirements

The following knowledges and environment are required to run this code:

* Python 3.7
* PyQt5
* Python librairies: PIL, OpenSlide, Numpy, CV2, Scikit-Image, CSV
* IDE, for instance PyCharm, VSCode with Python module

### Installation

Recommended step by step is:

* Prepare the Python environement
  * creating a specific venv is a good idea (see Python best practices on the Internet
* Clone the repository
* Run the Python file **gui.py**
  * the two other .py files are called by the main code gui.py

## Build with

Products below where used to build this application:

* [Anaconda](https://www.anaconda.com/products/individual) - Python environment
* [PyQt5](https://pypi.org/project/PyQt5/) - PyQt5
* [Visual Studio Code](https://code.visualstudio.com/docs/languages/markdown) - VS Code with modules Python for the program and markdown to create this README.md

## Versions

Available versions:

* **Last version :** 1.0

## Author

Application developed by:

* **Alexandre Cornier** _alias_ [@alexcornier](https://github.com/alexcornier/)

## License

This project is under ``MIT License`` License - Cf. file [LICENSE.md](LICENSE.md) for any further details.
