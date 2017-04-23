using System.Windows;


namespace ASD.CellUniverse {

    using Infrastructure.Properties;

    public partial class App : Application {

        public App() {
            Exit += (s, e) => Settings.Default.Save();
        }
    }
}