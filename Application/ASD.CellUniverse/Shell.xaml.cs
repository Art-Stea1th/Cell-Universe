using System.Windows;

namespace ASD.CellUniverse {

    using ViewModels;

    public partial class Shell : Window {
        public Shell() {
            InitializeComponent();
            DataContext = new ShellViewModel();
        }
    }
}