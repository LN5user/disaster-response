
import plotly.graph_objects as go



def return_figures(df):
    """
    Prepare and visualize data
    INPUT
    df: pandas dataframe

    OUTPUT
        None
    """

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    request_counts = df[df['request'] == 1].groupby('genre').count()['message']
    offer_counts = df[df['offer'] == 1].groupby('genre').count()['message']
   

    graph_one =[
                go.Bar(
                    x=genre_names,
                    y=genre_counts,
                    name = 'Total'

                ),
                go.Bar(
                    x=genre_names,
                    y= request_counts,
                    name = 'Request'
                ),
                go.Bar(
                    x=genre_names,
                    y= offer_counts,
                    name = 'Offer'
                )
            ]
        

    

    layout_one = dict(title='Distribution of Message Genres and Help Type',
                      xaxis=dict(title="Count"),
                      yaxis=dict(title="Genre"),
                      height=500,
                      width=1400,
                      autosize=False
                      )

    # count number of occurrences 1 for each label
    count_one_occurence = {}
    for col_name in df.columns[4:]:
        cnt = df[df[col_name] == 1].shape[0]
        col_name = col_name.replace('_', ' ')
        count_one_occurence[col_name] = cnt
    count_one_occurence = dict(
        sorted(count_one_occurence.items(), key=lambda item: item[1], reverse=True))
    graph_two = []

    graph_two.append(
        go.Bar(
            x=list(count_one_occurence.keys()),
            y=list(count_one_occurence.values()),

        )
    )

    layout_two = dict(title='Distribution of Disaster Types',
                      yaxis=dict(title="Count"),
                      xaxis=dict(title="Labels"),
                      height=500,
                      width=1400,
                      autosize=False
                      )

    graph_three = []
    graph_three .append(
        go.Pie(
            labels=list(count_one_occurence.keys()),
            values=list(count_one_occurence.values()),

        )
    )

    layout_three = dict(title='Distribution of Disaster Types in Percent ',
                        yaxis=dict(title="Count"),
                        xaxis=dict(title="Labels"),
                        height=900,
                        width=1200,
                        autosize=False
                        )

    # append all charts
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    #figures.append(dict(data=graph_four, layout=layout_four))
    return figures
