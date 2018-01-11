import os

from homyd.frame.learningtable import LearningTable

from csxdata.visual import Scatter2D


lt = LearningTable.from_xlsx(os.path.expanduser("~/SciProjects/Project_Fruits/adat.xlsx"),
                             labels="FAMILIA", paramset=["DH1", "D13C"])
lt.dropna(inplace=True)

scat = Scatter2D(lt.X, lt.Y, title="TestEm", axlabels=lt.paramset)
scat.split_scatter(show=True)
