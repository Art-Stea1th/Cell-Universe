using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;

namespace CellUniverse.Views.Styles.UWP.Dark.Window {

    public partial class DarkUWPWindowStyle {

        private const string WindowContainerName = "PART_WindowContainer";
        private const string WindowBorderName = "PART_WindowBorder";
        private System.Windows.Window window = null;

        void IconMouseLeftButtonDown(object sender, MouseButtonEventArgs e) {
            if (e.ClickCount > 1)
                sender.ForWindowFromTemplate(w => SystemCommands.CloseWindow(w));
        }

        void IconMouseUp(object sender, MouseButtonEventArgs e) {
            var element = sender as FrameworkElement;
            var point = element.PointToScreen(new Point(element.ActualWidth / 2, element.ActualHeight));
            sender.ForWindowFromTemplate(w => SystemCommands.ShowSystemMenu(w, point));
        }

        void WindowLoaded(object sender, RoutedEventArgs e) {
            window = (System.Windows.Window)sender;
            window.StateChanged += WindowStateChanged;
        }

        void WindowStateChanged(object sender, EventArgs e) {
            var handle = window.GetWindowHandle();
            var containerBorder = (Border)window.Template.FindName(WindowContainerName, window);

            if (window.WindowState == WindowState.Maximized) {
                var screen = System.Windows.Forms.Screen.FromHandle(handle);
                if (screen.Primary) {
                    containerBorder.Padding = new Thickness(
                        SystemParameters.WorkArea.Left + 9,
                        SystemParameters.WorkArea.Top + 9,
                        SystemParameters.PrimaryScreenWidth - SystemParameters.WorkArea.Right,
                        SystemParameters.PrimaryScreenHeight - SystemParameters.WorkArea.Bottom);
                }
            }
            else {
                containerBorder.Padding = new Thickness(7, 7, 7, 7);
            }
        }        

        void MinimizeButtonClick(object sender, RoutedEventArgs e) {
            sender.ForWindowFromTemplate(w => SystemCommands.MinimizeWindow(w));
        }

        void MaximizeButtonClick(object sender, RoutedEventArgs e) {
            sender.ForWindowFromTemplate(w => {
                if (w.WindowState == WindowState.Maximized) SystemCommands.RestoreWindow(w);
                else SystemCommands.MaximizeWindow(w);
            });
        }

        void CloseButtonClick(object sender, RoutedEventArgs e) {
            sender.ForWindowFromTemplate(w => SystemCommands.CloseWindow(w));
        }
    }
}