using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace CellUniverse.Views.Styles.UWP.Dark.Window {

    internal static class LocalExtensions {

        public static void ForTemplatedWindow(this object templateFrameworkElement, Action<System.Windows.Window> action) {
            System.Windows.Window window = ((FrameworkElement)templateFrameworkElement).TemplatedParent as System.Windows.Window;
            if (window != null) action(window);
        }
    }

    public partial class DarkUWPWindowStyle {

        private const string WindowContainerName = "PART_WindowContainer";
        private const string WindowBorderName = "PART_WindowBorder";
        private System.Windows.Window window = null;

        void IconMouseLeftButtonDown(object sender, MouseButtonEventArgs e) {
            if (e.ClickCount > 1)
                sender.ForTemplatedWindow(w => w.Close());
        }

        void IconMouseUp(object sender, MouseButtonEventArgs e) {
            var element = sender as FrameworkElement;
            var point = element.PointToScreen(new Point(element.ActualWidth / 2, element.ActualHeight));
            sender.ForTemplatedWindow(w => SystemCommands.ShowSystemMenu(w, point));
        }

        void WindowLoaded(object sender, RoutedEventArgs e) {
            window = (System.Windows.Window)sender;
        }      

        void MinimizeButtonClick(object sender, RoutedEventArgs e) {
            sender.ForTemplatedWindow(w => w.WindowState = WindowState.Minimized);
        }

        void MaximizeButtonClick(object sender, RoutedEventArgs e) {
            sender.ForTemplatedWindow(w => {

                var containerBorder = (Border)window.Template.FindName(WindowContainerName, window);

                if (w.WindowState == WindowState.Maximized) {
                    w.WindowState = WindowState.Normal;
                    containerBorder.Padding = new Thickness(7, 7, 7, 7);
                }
                else {
                    w.WindowState = WindowState.Maximized;                    
                    containerBorder.Padding = new Thickness(
                        SystemParameters.WorkArea.Left + 6,
                        SystemParameters.WorkArea.Top + 6,
                        SystemParameters.PrimaryScreenWidth - SystemParameters.WorkArea.Right + 7,
                        SystemParameters.PrimaryScreenHeight - SystemParameters.WorkArea.Bottom + 6);
                }
            });
        }

        void CloseButtonClick(object sender, RoutedEventArgs e) {
            sender.ForTemplatedWindow(w => w.Close());
        }
    }
}