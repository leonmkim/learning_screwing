## for renaming group names of runs
api = Api()
run = api.run(run_name)
run.group = new_group_name
run.update()
