using System;
using System.Windows.Media;
using System.Windows.Controls;
using System.Windows.Threading;

namespace GameOfLife.Views.Controls {

    using ViewModels;

    public partial class CellUniversePanel : UserControl {

        private CellUniverseViewModel _cellUniverseViewModel;

        public CellUniversePanel() {
            InitializeComponent();
            _cellUniverseViewModel = DataContext as CellUniverseViewModel;
            StartGame();
        }

        private void StartGame() {
            DispatcherTimer timer = new DispatcherTimer();
            timer.Interval = TimeSpan.FromMilliseconds(1);
            timer.Tick += (s, e) => { Update(); };
            timer.Start();
        }

        public void Update() {
            _cellUniverseViewModel.ViewRecalculate((int) ActualWidth, (int)ActualHeight);
            cellUniverseImage.Source = _cellUniverseViewModel.CellUniverseWritableBitmap;
            cellUniverseImage.Stretch = Stretch.None;
            _cellUniverseViewModel.Next();
        }
    }
}