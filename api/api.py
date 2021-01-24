import twint
import json
import flask
from flask import request, jsonify

COVID = "2020-03-11" # WHO declared COVID-19 a pandemic on March 11 2020 [https://en.wikipedia.org/wiki/COVID-19_pandemic]

app = flask.Flask(__name__)

@app.route('/api/scrape', methods=['GET'])
def api_scrape():
    if 'query' in request.args:
        query = str(request.args['query'])
    else:
        return "Error: No query field provided. Please specify an query."

    before = scrape_before(query)

    after = scrape_after(query)

    data = { 'before' : before, 'after': after }

    return(data)

def scrape_before(query):

    open('before.json', 'w').truncate(0) # placeholder to reset file

    c = twint.Config()
    c.Hide_output = True

    c.Search = query
    c.Until = COVID

    c.Limit = 1  # placeholder to limit tweets

    c.Store_json = True
    c.Output = "before.json"

    twint.run.Search(c)

    return(open('before.json', 'r').read())

def scrape_after(query):

    open('after.json', 'w').truncate(0) # placeholder to reset file

    c = twint.Config()
    c.Hide_output = True

    c.Search = query
    c.Since = COVID

    c.Limit = 1  # placeholder to limit tweets

    c.Store_json = True
    c.Output = "after.json"

    twint.run.Search(c)

    return(open('after.json', 'r').read())