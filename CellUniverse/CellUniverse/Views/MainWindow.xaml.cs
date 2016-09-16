using System.Windows;


namespace CellUniverse.Views {

    using ViewModels;

    public partial class MainWindow : Window {

        private MainWindowViewModel _mainWindowViewModel;

        public MainWindow() {
            InitializeComponent();
            _mainWindowViewModel = new MainWindowViewModel(cellUniverseSurface);
            DataContext = _mainWindowViewModel;
        }

        private void CellUniverseSurfaceSizeChanged(object sender, SizeChangedEventArgs e) {
            _mainWindowViewModel.CellSurfaceInvalidate();
        }
    }
}