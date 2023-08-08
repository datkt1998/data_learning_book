
from dash import Dash, html, dcc, Input, Output
import pandas as pd
import plotly.express as px

# data
happiness = pd.read_csv('data/13 - worldhappiness.csv')
line_fg = px.line(                                                  # add line plot
    happiness[happiness['country'] == 'United States'],
    x = 'year',
    y = 'happiness_score',
    title='Happiness Score in United States'
)


# html Component
#     section headings {H1, H2, ...}
#     paragraph {p} (text noi dung), 
#     content division {Div} ( vung chua noi dung), 
#     line-break {br}, 
#     anchor{a} ( gan hyperlink), 
#     html tags 

app = Dash()
app.layout = html.Div(children=[
    html.H1('The Sample Dashboard'),                                # header 1
    html.P([                                                        # paragraph
        'This is the sample dashboard display the scores',              # text
        html.Br(),                                                      # line break
        html.A(                                                         # hyperlink
            'World happiness Report Data Source',                           # display
            href='http://www.worldhappiness.report',                        # link
            target= '_blank'                                                # open new tab when clicked link
            ),

    dcc.RadioItems(                                                 # select ONE item in current
        options = happiness['region'].unique() ,                        # list/dict of option values can be choose
        value= 'North America'                                          # first item selected when open
        ),
    dcc.Checklist(                                                 # select MULTI-items in current
        options = happiness['region'].unique() ,                        # list/dict of option values can be choose
        value= ['North America']                                        # first items selected when open
        ),
    dcc.Dropdown(                                                  # select from dropdown list items   
        options = happiness['country'].unique() ,                       # list/dict of option values can be choose
        value= ['United States']                                        
        ),

    dcc.Graph(                                                      # add graph component 
        figure =  line_fg                                          
        )
    ])
])





if __name__ == '__main__':
    app.run_server(debug=True) # debug thuong dung khi test, reload web se update theo the newest code. Neu moi truowng prod thi nen set debug = False