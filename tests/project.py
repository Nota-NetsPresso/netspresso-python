from netspresso import NetsPresso

EMAIL = "bmlee@nota.ai"
PASSWORD = "Sdlqudaks12!"

netspresso = NetsPresso(email=EMAIL, password=PASSWORD)

project_name = "project_1217_5"
project = netspresso.create_project(project_name=project_name)
print(project.project_name)
