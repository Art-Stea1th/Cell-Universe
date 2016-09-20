using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Threading;

namespace RCVL.IntegrationTests {

    public class MainWindowViewModel : ViewModelBase {

        private Random   _random;
        private Color[,] _cellularData;

        public Color[,] CellularData {
            get {
                return _cellularData;
            }
            set {
                _cellularData = value;
                OnPropertyChanged(GetMemberName((MainWindowViewModel c) => c.CellularData));
            }
        }

        public MainWindowViewModel() {
            _random = new Random();
            Start();
        }        

        private void Start() {
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(10);
            timer.Tick += (s, e) => { Update(); };
            timer.Start();
        }

        private void Update() {

            int width = 160;
            int height = 90;

            Color[,] TrueRedRandom   = GetRandomLayer(width, height, Color.FromRgb(255, 0, 0));
            Color[,] TrueGreenRandom = GetRandomLayer(width, height, Color.FromRgb(0, 255, 0));
            Color[,] TrueBlueRandom  = GetRandomLayer(width, height, Color.FromRgb(0, 0, 255));

            Color[,] tempData = new Color[height, width];

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    tempData[y, x] += TrueRedRandom[y, x];
                    tempData[y, x] += TrueGreenRandom[y, x];
                    tempData[y, x] += TrueBlueRandom[y, x];
                }
            }
            CellularData = tempData;
        }

        private Color[,] GetRandomLayer(int width, int height, Color color) {

            Color[,] result = new Color[height, width];

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    result[y, x] = _random.Next(2) == 1 ? color : Color.FromRgb(0, 0, 0);
                }
            }
            return result;
        }
    }
}