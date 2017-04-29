using System.Windows.Data;


namespace ASD.CellUniverse.Infrastructure.BindingExtensions {

    using Properties;

    public class SettingsExtension : Binding {

        public SettingsExtension() => Initialize();
        public SettingsExtension(string path) : base(path) => Initialize();

        private void Initialize() {
            Source = Settings.Default;
            Mode = BindingMode.TwoWay;
        }
    }
}