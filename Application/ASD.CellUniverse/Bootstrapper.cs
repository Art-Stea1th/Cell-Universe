using System.Windows;
using Microsoft.Practices.Unity;
using Prism.Unity;

namespace ASD.CellUniverse {

    internal sealed class Bootstrapper : UnityBootstrapper {

        protected override DependencyObject CreateShell() => Container.Resolve<Shell>();
        protected override void InitializeShell() => Application.Current.MainWindow.Show();
    }
}