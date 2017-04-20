using System.Windows;


namespace ASD.CellUniverse {

    using Infrastructure.Properties;

    public partial class App : Application {

        public App() {

            Startup += (s, e) => {
                var bootstrapper = new Bootstrapper();
                bootstrapper.Run();
            };

            Exit += (s, e) => Settings.Default.Save();
        }
    }
}