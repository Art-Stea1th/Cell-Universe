using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace ASD.CellUniverse.Resources.Controls {

    public partial class WindowBase {

        private const string windowPaddingName = "PART_WindowPadding";
        private Window window = null;

        private object stateLocker = new object();
        private bool inChaging = false;

        private void IconMouseLeftButtonDown(object sender, MouseButtonEventArgs e) {
            if (e.ClickCount > 1) {
                sender.ForTemplatedWindow(w => w.Close());
            }
        }

        private void CaptionMouseLeftButtonDown(object sender, MouseButtonEventArgs e) {
            if (e.ClickCount > 1) {
                MaximizeButtonClick(sender, e);
            }
        }

        private void IconMouseUp(object sender, MouseButtonEventArgs e) {
            var element = sender as FrameworkElement;
            var point = element.PointToScreen(new Point(element.ActualWidth / 2, element.ActualHeight));
            sender.ForTemplatedWindow(w => SystemCommands.ShowSystemMenu(w, point));
        }

        private void WindowLoaded(object sender, RoutedEventArgs e) {
            window = (Window)sender;
            window.StateChanged += WindowStateChanged;
        }

        private void WindowStateChanged(object sender, EventArgs e) {
            var containerBorder = (Border)window.Template.FindName(windowPaddingName, window);

            if (window.WindowState == WindowState.Maximized && !inChaging) {

                lock (stateLocker) {
                    inChaging = true;
                    window.WindowState = WindowState.Normal;
                    window.WindowState = WindowState.Maximized;
                    inChaging = false;
                }
                containerBorder.Padding = new Thickness(
                        SystemParameters.WorkArea.Left + 7,
                        SystemParameters.WorkArea.Top + 7,
                        SystemParameters.PrimaryScreenWidth - SystemParameters.WorkArea.Right + 7,
                        SystemParameters.PrimaryScreenHeight - SystemParameters.WorkArea.Bottom + 7);
            }
            else {
                containerBorder.Padding = new Thickness(7);
            }
        }

        private void MinimizeButtonClick(object sender, RoutedEventArgs e) {
            sender.ForTemplatedWindow(w => w.WindowState = WindowState.Minimized);
        }

        private void MaximizeButtonClick(object sender, RoutedEventArgs e) {
            sender.ForTemplatedWindow(w => {
                if (w.WindowState == WindowState.Maximized) { w.WindowState = WindowState.Normal; }
                else { w.WindowState = WindowState.Maximized; }
                e.Handled = true;
            });
        }

        private void CloseButtonClick(object sender, RoutedEventArgs e) {
            sender.ForTemplatedWindow(w => w.Close());
        }
    }

    internal static class LocalExtensions {

        public static void ForTemplatedWindow(this object templateFrameworkElement, Action<Window> action) {
            if (((FrameworkElement)templateFrameworkElement).TemplatedParent is Window window) {
                action(window);
            }
        }
    }
}