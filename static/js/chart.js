class Chart {
    constructor() {
        this.height = 350;
        this.chartObject = new Object();
        this.isLoading = true;

        axios.defaults.xsrfCookieName = 'csrftoken';
		axios.defaults.xsrfHeaderName = 'X-CSRFToken';
    }

    drawChart(canvas_id, data, type, colors, height=null) {
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
            xaxis: {
                type: 'datetime',
                labels: {
                  format: 'yyyy MM'
                }
            },
        };

        const chart = new ApexCharts(document.getElementById(canvas_id), options);
        chart.render();
        return chart;
    }

    updateChart(chart, data) {
        chart.updateSeries(data);
    }

    progressLoading(status) {
        this.isLoading = status;
        const progressLoading = document.querySelector('.progress-loading');

        if (this.isLoading == false) {
            progressLoading.classList.add('d-none');
        } else {
            progressLoading.classList.remove('d-none');
        }
    }

    getAxios(url, method, formData=null) {
        const self = this;
        let kwargs = {
            url: url,
	        method: method,
        }

        if (method == 'post' && formData != null) {
            kwargs['data'] = formData;
        }

        axios(kwargs)
        .then(function (response) {
            const data = response.data;

            console.log(data);

            Object.keys(data).forEach(key => {
                if (key !== 'metric') {
                    if (method == 'get') {
                        self.chartObject[key] = self.drawChart(
                            key, 
                            data[key]['data'], 
                            data[key]['type'],
                            data[key]['colors'],
                            data[key]['height']
                        );
                    } else {
                        self.updateChart(
                            self.chartObject[key],
                            data[key]['data'], 
                        );
                    }
                } else {
                    // TODO: method로 분리 필요 및 어떤 Universe인지 확인할 구분값 넘겨 받아서 if 문처리 필요
                    Object.keys(data['metric']).forEach(key => {
                        if (key !== 'returns' && key !== 'CAGR') {
                            document.querySelectorAll(`#${key} .metric-value`).forEach((ele, idx) => {
                                ele.innerText = data['metric'][key][idx]['data'];
                                ele.style.color = data['metric'][key][idx]['color'];
                            });
                        } else {
                            data['metric'][key].forEach(e => {
                                const ele = document.getElementById(`${e['name']}-${key}`);
                                ele.innerText = `${e['data']}%`;
                                ele.style.color = e['color'];
                            })
                        }
                    });
                }
            })
        }).catch(function (err) {
            console.error(err);
        }).finally(function () {
            self.progressLoading(false);
        });
    }

    requestGet(url) {
        this.getAxios(url, 'get');
    }

    requestPost(url) {
        document.getElementById('select-options-form').addEventListener('submit', e => {
            e.preventDefault();

            const formData = new FormData(e.target);

            this.progressLoading(true);
            this.getAxios(url, 'post', formData);
        });
    }
}