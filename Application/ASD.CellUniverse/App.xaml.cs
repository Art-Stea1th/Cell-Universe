using System;
using System.Windows;


namespace ASD.CellUniverse {

    using Infrastructure.Properties;
    using ViewModels;

    public partial class App : Application {

        public App() {

            IDisposable disposableViewModel = null;

            Startup += (sender, args) => {
                MainWindow = new Shell() { DataContext = new ShellViewModel() };
                disposableViewModel = MainWindow.DataContext as IDisposable;
                MainWindow.Show();
            };

            DispatcherUnhandledException += (s, e) => {
                disposableViewModel?.Dispose();
            };

            Exit += (s, e) => {
                disposableViewModel?.Dispose();
                Settings.Default.Save();
            };
        }
    }
}