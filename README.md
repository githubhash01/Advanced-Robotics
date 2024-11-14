# HOW TO USE THE REPO 

You can replicate the figures and tables in the report by going to test.py.

You can follow the `TODO MARKER` to find what parameters to change to run the tests.
The plots will be generated in the `images\` folder. If it's a table, the output will be in the terminal.

In SUMMARY, to run a test, follow the steps below:
1) Open test.py
2) Change the `TEST_TO_RUN` variable to the test you want to run by looking at the table above the variable, under the column `TEST_TO_RUN`.
3) To change the test scenario, such as `Standard`, `Reversed`, `Rotated Start`..., go to `config.py` and uncomment the desired `TEST_CASE`.
4) run `python3 test.py` in the terminal.
5) You can find the generated plots in `images\` folder. If it's a table, the output will be in the terminal.
