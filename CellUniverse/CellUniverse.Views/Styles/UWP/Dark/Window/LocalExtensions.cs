using System;
using System.Windows;
using System.Windows.Interop;


namespace CellUniverse.Views.Styles.UWP.Dark.Window {

    internal static class LocalExtensions {

        public static void ForWindowFromTemplate(this object templateFrameworkElement, Action<System.Windows.Window> action) {
            System.Windows.Window window = ((FrameworkElement)templateFrameworkElement).TemplatedParent as System.Windows.Window;
            if (window != null) action(window);
        }

        public static IntPtr GetWindowHandle(this System.Windows.Window window) {
            WindowInteropHelper helper = new WindowInteropHelper(window);
            return helper.Handle;
        }
    }
}