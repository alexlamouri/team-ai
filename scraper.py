import twint
import json

# WHO declared COVID-19 a pandemic on March 11 2020 [https://en.wikipedia.org/wiki/COVID-19_pandemic]
COVID = "2020-03-11"

search_term = input("Enter a search term: ")

# Before COVID-19
c = twint.Config()
c.Hide_output = True

c.Search = search_term
c.Until = COVID

c.Store_json = True
c.Output = "before.json"

c.Limit = 1  # placeholder

twint.run.Search(c)

# After COVID-19
c = twint.Config()
c.Hide_output = True

c.Search = search_term
c.Until = COVID

c.Store_json = True
c.Output = "after.json"

c.Limit = 1  # placeholder

twint.run.Search(c)
