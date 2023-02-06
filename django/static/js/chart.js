class Chart {
    constructor() {
        this.height = 350;
    }

    draw(canvas_id, data, type, colors, height=null) {
        const options = {
            series: data,

            colors: colors,

            chart: {
                type: type,
                stacked: false,
                height: (height === null) ? this.height : height,
                width: '100%',
                zoom: {
                    type: 'xy',
                    enabled: true,
                    autoScaleYaxis: true
                },
                toolbar: {
                    autoSelected: 'zoom'
                }
            },

            dataLabels: {
                enabled: false
            },

            markers: {
                size: 0,
            },

            fill: {
                type: 'gradient',
                gradient: {
                    opacityFrom: 0,
                    opacityTo: 0,
                },
            },

            tooltip: {
                shared: false,
            },
        };

        const chart = new ApexCharts(document.getElementById(canvas_id), options);
        chart.render();
    }
}