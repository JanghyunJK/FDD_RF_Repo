import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def fault_impact_sum(df_impact, configs):

    #------------------------------------------------------------#
    # plotting
    #------------------------------------------------------------#
    title_font_size = 12
    tick_font_size = 12
    fontfamily = 'Times New Roman'
    barwidth = 0.75
    colorscale = ['rgb(215,25,28)','rgb(253,174,97)','rgb(171,221,164)','rgb(43,131,186)']

    fig = make_subplots(
        rows=1, 
        cols=4, 
        shared_yaxes=True, 
        horizontal_spacing=0.05,
    )  

    list_plot = [
        'fault_duration_ratio',
        'impact_site_energy_elec_kWh',
        'impact_site_energy_ng_kWh',
        'impact_cost_$'
    ]

    list_title_x = [
        "<b>Diagnosis<br>ratio [-]</b>",
        "<b>Excess<br>electricity [kWh]</b>",
        "<b>Excess<br>natural gas [kWh]</b>",
        "<b>Excess<br>cost [$]</b>"
    ]

    #------------------------------------------------------------#
    # bar plots
    #------------------------------------------------------------#
    col=0

    for plot in list_plot:
        
        fig.add_trace(go.Bar(
            x=df_impact[plot],
            y=df_impact.fdd_result,
            text=round(df_impact[plot],2),
            textposition='auto',
            textfont_family=fontfamily,
            orientation='h',
            marker=dict(
                color=colorscale[col],
                line=dict(color='black', width=1)
            ),
            showlegend=False,
            width=barwidth,
        ),row=1,col=col+1)
        
        col+=1

    #------------------------------------------------------------#
    # axes
    #------------------------------------------------------------#
    col=1

    for title in list_title_x:

        fig.update_xaxes(
            title = dict( 
                text=title,
                font=dict(
                    family=fontfamily,
                    size=title_font_size,
                ),
            ),
            tickfont = dict(
                family=fontfamily,
                size=tick_font_size,
            ),
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='black',
            row=1, col=col
        )
        
        col+=1
        
    fig.update_yaxes(
        tickfont = dict(
            family=fontfamily,
            size=tick_font_size,
        ),
    )

    #------------------------------------------------------------#
    # axes
    #------------------------------------------------------------#
    fig.update_layout(
        width=800,
        height=400,
        margin=dict(
            l=200,
            r=0,
            t=0,
            b=50,
        ),
        plot_bgcolor='white',
    )

    # export
    path_impact_visual = configs['dir_results'] + "/{}_FDD_impact_sum.svg".format(configs["weather"])
    print("[Estimating Fault Impact] saving fault impact estimation figure in {}".format(path_impact_visual))
    pio.write_image(fig, path_impact_visual)


def fault_impact_heatmap_power(df_combined, configs):

    # calculate monthly and annual excess energy usages
    if (configs['sensor_unit_ng']=='W') & (configs['sensor_unit_elec']=='W'):
        df_monthly = df_combined.groupby(['Month'])[["baseline_elec_{}".format(configs["sensor_unit_elec"]),"baseline_ng_{}".format(configs["sensor_unit_ng"]),'diff_elec','diff_ng']].sum()/1000/(60/configs['impact_est_timestep_min']) #convert W to kWh
        base_annual_elec = round(df_monthly["baseline_elec_{}".format(configs["sensor_unit_elec"])].sum()) # in kWh
        base_annual_ng = round(df_monthly["baseline_ng_{}".format(configs["sensor_unit_ng"])].sum()) # in kWh
        diff_annual_elec = round(df_monthly.sum()['diff_elec']) # in kWh
        diff_annual_ng = round(df_monthly.sum()['diff_ng']) # in kWh
        perc_annual_elec = round(diff_annual_elec/base_annual_elec*100, 3) # in %
        perc_annual_ng = round(diff_annual_ng/base_annual_ng*100, 3) # in %
    else:
        # add other unit conversions
        print("[Estimating Fault Impact] unit conversion from {} for electricity and {} for natural gas to kWh is not currently supported".format(configs['sensor_unit_elec'],configs['sensor_unit_ng']))

    # plot setting
    title_font_size = 12
    colorbar_font_size = 12
    tick_font_size = 12
    anot_font_size = 16
    fontfamily = 'Times New Roman'
    colorscale=[
        [0.0, 'rgb(5,48,97)'],
        [0.1, 'rgb(33,102,172)'],
        [0.2, 'rgb(67,147,195)'],
        [0.3, 'rgb(146,197,222)'],
        [0.4, 'rgb(209,229,240)'],
        [0.5, 'rgb(247,247,247)'],
        [0.6, 'rgb(253,219,199)'],
        [0.7, 'rgb(244,165,130)'],
        [0.8, 'rgb(214,96,77)'],
        [0.9, 'rgb(178,24,43)'],
        [1.0, 'rgb(103,0,31)']
    ]
    # colorscale=[
    #     [0.0, 'rgb(49,54,149)'],
    #     [0.1, 'rgb(69,117,180)'],
    #     [0.2, 'rgb(116,173,209)'],
    #     [0.3, 'rgb(171,217,233)'],
    #     [0.4, 'rgb(224,243,248)'],
    #     [0.5, 'rgb(255,242,204)'],
    #     [0.6, 'rgb(254,224,144)'],
    #     [0.7, 'rgb(253,174,97)'],
    #     [0.8, 'rgb(244,109,67)'],
    #     [0.9, 'rgb(215,48,39)'],
    #     [1.0, 'rgb(165,0,38)']
    # ]
    range_max_elec = max( df_combined['diff_elec'].max() , abs(df_combined['diff_elec'].min()) )
    range_max_ng = max( df_combined['diff_ng'].max() , abs(df_combined['diff_ng'].min()) )

    # plotting
    num_rows = 2
    num_cols = 2
    fig = make_subplots(
        rows=num_rows, 
        cols=num_cols, 
        shared_xaxes=True, 
        vertical_spacing=0.025,
        horizontal_spacing=0.1,
        column_widths=[0.1, 0.4],
    )  

    # heatmap
    fig.add_trace(go.Heatmap(
        z=df_combined['diff_elec'],
        x=df_combined['Date'],
        y=df_combined['Time'],
        colorscale='tempo',
        coloraxis='coloraxis1',
    ),
    row=1, col=2)
    fig.add_trace(go.Heatmap(
        z=df_combined['diff_ng'],
        x=df_combined['Date'],
        y=df_combined['Time'],
        colorscale='tempo',
        coloraxis='coloraxis2',
    ),
    row=2, col=2)

    # annotation
    if perc_annual_elec > 0:
        text_elec = "+"
    else:
        text_elec = ""

    if perc_annual_ng > 0:
        text_ng = "+"
    else:
        text_ng = ""
    fig.add_annotation(
        x=0.08,
        y=0.75,
        xref="paper",
        yref="paper",
        xanchor='center',
        yanchor='middle',
        text="Excess<br>electricity<br><b>{} kWh<br>({}{}%)</b>".format(diff_annual_elec, text_elec, perc_annual_elec),
        font=dict(
            family=fontfamily,
            size=anot_font_size,
            ),
        showarrow=False,
        align="right",
        )
    fig.add_annotation(
        x=0.08,
        y=0.25,
        xref="paper",
        yref="paper",
        xanchor='center',
        yanchor='middle',
        text="Excess<br>natural gas<br><b>{} kWh<br>({}{}%)</b>".format(diff_annual_ng, text_ng, perc_annual_ng),
        font=dict(
            family=fontfamily,
            size=anot_font_size,
            ),
        showarrow=False,
        align="right",
        )

    # layout
    fig.update_layout(
        width=800,
        height=400,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        plot_bgcolor='white',
        coloraxis1=dict(
            cmin=-range_max_elec,
            cmid=0,
            cmax=range_max_elec,
            colorscale=colorscale, 
            colorbar = dict(
                title=dict(
                    text = "Excess electricty [{}]".format(configs['sensor_unit_elec']),
                    side='right',
                    font=dict(
                        size=colorbar_font_size,
                        family=fontfamily,
                    ),
                ),
                len=0.5,
                x=1,
                xanchor='left',
                y=0.75,
                yanchor='middle',
                thickness=23,
            )
        ),
        coloraxis2=dict(
            cmin=-range_max_ng,
            cmid=0,
            cmax=range_max_ng,
            colorscale=colorscale, 
            colorbar = dict(
                title=dict(
                    text = "Excess natural gas [{}]".format(configs['sensor_unit_ng']),
                    side='right',
                    font=dict(
                        size=colorbar_font_size,
                        family=fontfamily,
                    ),
                ),
                len=0.5,
                x=1,
                xanchor='left',
                y=0.25,
                yanchor='middle',
                thickness=23,
            ),
        ),
    )

    # axes
    for row in range(1, num_rows+1):
        for col in range(1, num_cols+1):
            if col==1:
                fig.update_yaxes(
                    showticklabels=False,
                    row=row, col=col
                )      
            elif col==2:     
                fig.update_yaxes(
                    tickfont = dict(
                        family=fontfamily,
                        size=tick_font_size,
                    ),
                    row=row, col=col
                )
            elif col==3:
                fig.update_yaxes(
                    title = dict( 
                        text="<b>Time</b>",
                        font=dict(
                            family=fontfamily,
                            size=title_font_size,
                        ),
                        standoff=0,
                    ),
                    tickfont = dict(
                        family=fontfamily,
                        size=tick_font_size,
                    ),
                    row=row, col=col
                )      

    fig.update_xaxes(
        title = dict( 
            text="<b>Date</b>",
            font=dict(
                family=fontfamily,
                size=title_font_size,
            ),
        ),
        tickfont = dict(
            family=fontfamily,
            size=tick_font_size,
        ),
        tickformat="%b",
        dtick="M2",
        row=2, col=3
    )
    fig.update_xaxes(
        title = dict( 
            text="<b>Month</b>",
            font=dict(
                family=fontfamily,
                size=title_font_size,
            ),
        ),
        tickfont = dict(
            family=fontfamily,
            size=tick_font_size,
        ),
        row=2, col=2
    )
    fig.update_xaxes(
        dtick=1,
        row=1, col=2
    )

    # export
    path_impact_visual = configs['dir_results'] + "/{}_FDD_impact_figure_heatmap_power.svg".format(configs["weather"])
    print("[Estimating Fault Impact] saving fault impact estimation figure in {}".format(path_impact_visual))
    pio.write_image(fig, path_impact_visual)


def fault_impact_heatmap_cost(df_combined, configs):

    df_monthly = df_combined.groupby(['Month'])[["baseline_elec_demand_cost_$","baseline_elec_energy_cost_$","baseline_ng_cost_$","diff_elec_cost_$","diff_ng_cost_$"]].sum()
    base_annual_elec = round( df_monthly["baseline_elec_demand_cost_$"].sum() + df_monthly["baseline_elec_energy_cost_$"].sum() ) # in $
    base_annual_ng = round(df_monthly["baseline_ng_cost_$"].sum()) # in $
    diff_annual_elec = round(df_monthly.sum()['diff_elec_cost_$']) # in $
    diff_annual_ng = round(df_monthly.sum()['diff_ng_cost_$']) # in $
    perc_annual_elec = round(diff_annual_elec/base_annual_elec*100, 3) # in %
    perc_annual_ng = round(diff_annual_ng/base_annual_ng*100, 3) # in %

    # plot setting
    title_font_size = 12
    colorbar_font_size = 12
    tick_font_size = 12
    anot_font_size = 16
    fontfamily = 'Times New Roman'
    colorscale=[
        [0.0, 'rgb(5,48,97)'],
        [0.1, 'rgb(33,102,172)'],
        [0.2, 'rgb(67,147,195)'],
        [0.3, 'rgb(146,197,222)'],
        [0.4, 'rgb(209,229,240)'],
        [0.5, 'rgb(247,247,247)'],
        [0.6, 'rgb(253,219,199)'],
        [0.7, 'rgb(244,165,130)'],
        [0.8, 'rgb(214,96,77)'],
        [0.9, 'rgb(178,24,43)'],
        [1.0, 'rgb(103,0,31)']
    ]
    # colorscale=[
    #     [0.0, 'rgb(49,54,149)'],
    #     [0.1, 'rgb(69,117,180)'],
    #     [0.2, 'rgb(116,173,209)'],
    #     [0.3, 'rgb(171,217,233)'],
    #     [0.4, 'rgb(224,243,248)'],
    #     [0.5, 'rgb(255,242,204)'],
    #     [0.6, 'rgb(254,224,144)'],
    #     [0.7, 'rgb(253,174,97)'],
    #     [0.8, 'rgb(244,109,67)'],
    #     [0.9, 'rgb(215,48,39)'],
    #     [1.0, 'rgb(165,0,38)']
    # ]
    range_max_elec = max( df_combined['diff_elec_cost_$'].max() , abs(df_combined['diff_elec_cost_$'].min()) )
    range_max_ng = max( df_combined['diff_ng_cost_$'].max() , abs(df_combined['diff_ng_cost_$'].min()) )

    # plotting
    num_rows = 2
    num_cols = 2
    fig = make_subplots(
        rows=num_rows, 
        cols=num_cols, 
        shared_xaxes=True, 
        vertical_spacing=0.025,
        horizontal_spacing=0.1,
        column_widths=[0.1, 0.4],
    )  

    # heatmap
    fig.add_trace(go.Heatmap(
        z=df_combined['diff_elec_cost_$'],
        x=df_combined['Date'],
        y=df_combined['Time'],
        colorscale='tempo',
        coloraxis='coloraxis1',
    ),
    row=1, col=2)
    fig.add_trace(go.Heatmap(
        z=df_combined['diff_ng_cost_$'],
        x=df_combined['Date'],
        y=df_combined['Time'],
        colorscale='tempo',
        coloraxis='coloraxis2',
    ),
    row=2, col=2)

    # annotation
    if perc_annual_elec > 0:
        text_elec = "+"
    else:
        text_elec = ""

    if perc_annual_ng > 0:
        text_ng = "+"
    else:
        text_ng = ""
    fig.add_annotation(
        x=0.08,
        y=0.75,
        xref="paper",
        yref="paper",
        xanchor='center',
        yanchor='middle',
        text="Excess<br>electricity<br><b>$ {}<br>({}{}%)</b>".format(diff_annual_elec, text_elec, perc_annual_elec),
        font=dict(
            family=fontfamily,
            size=anot_font_size,
            ),
        showarrow=False,
        align="right",
        )
    fig.add_annotation(
        x=0.08,
        y=0.25,
        xref="paper",
        yref="paper",
        xanchor='center',
        yanchor='middle',
        text="Excess<br>natural gas<br><b>$ {}<br>({}{}%)</b>".format(diff_annual_ng, text_ng, perc_annual_ng),
        font=dict(
            family=fontfamily,
            size=anot_font_size,
            ),
        showarrow=False,
        align="right",
        )

    # layout
    fig.update_layout(
        width=800,
        height=400,
        margin=dict(
            l=0,
            r=0,
            t=0,
            b=0,
        ),
        plot_bgcolor='white',
        coloraxis1=dict(
            cmin=-range_max_elec,
            cmid=0,
            cmax=range_max_elec,
            colorscale=colorscale, 
            colorbar = dict(
                title=dict(
                    text = "Excess electricty [$]",
                    side='right',
                    font=dict(
                        size=colorbar_font_size,
                        family=fontfamily,
                    ),
                ),
                len=0.5,
                x=1,
                xanchor='left',
                y=0.75,
                yanchor='middle',
                thickness=23,
            )
        ),
        coloraxis2=dict(
            cmin=-range_max_ng,
            cmid=0,
            cmax=range_max_ng,
            colorscale=colorscale, 
            colorbar = dict(
                title=dict(
                    text = "Excess natural gas [$]",
                    side='right',
                    font=dict(
                        size=colorbar_font_size,
                        family=fontfamily,
                    ),
                ),
                len=0.5,
                x=1,
                xanchor='left',
                y=0.25,
                yanchor='middle',
                thickness=23,
            ),
        ),
    )

    # axes
    for row in range(1, num_rows+1):
        for col in range(1, num_cols+1):
            if col==1:
                fig.update_yaxes(
                    showticklabels=False,
                    row=row, col=col
                )      
            elif col==2:     
                fig.update_yaxes(
                    tickfont = dict(
                        family=fontfamily,
                        size=tick_font_size,
                    ),
                    row=row, col=col
                )
            elif col==3:
                fig.update_yaxes(
                    title = dict( 
                        text="<b>Time</b>",
                        font=dict(
                            family=fontfamily,
                            size=title_font_size,
                        ),
                        standoff=0,
                    ),
                    tickfont = dict(
                        family=fontfamily,
                        size=tick_font_size,
                    ),
                    row=row, col=col
                )      

    fig.update_xaxes(
        title = dict( 
            text="<b>Date</b>",
            font=dict(
                family=fontfamily,
                size=title_font_size,
            ),
        ),
        tickfont = dict(
            family=fontfamily,
            size=tick_font_size,
        ),
        tickformat="%b",
        dtick="M2",
        row=2, col=3
    )
    fig.update_xaxes(
        title = dict( 
            text="<b>Month</b>",
            font=dict(
                family=fontfamily,
                size=title_font_size,
            ),
        ),
        tickfont = dict(
            family=fontfamily,
            size=tick_font_size,
        ),
        row=2, col=2
    )
    fig.update_xaxes(
        dtick=1,
        row=1, col=2
    )

    # export
    path_impact_visual = configs['dir_results'] + "/{}_FDD_impact_figure_heatmap_cost.svg".format(configs["weather"])
    print("[Estimating Fault Impact] saving fault impact estimation figure in {}".format(path_impact_visual))
    pio.write_image(fig, path_impact_visual)
    
