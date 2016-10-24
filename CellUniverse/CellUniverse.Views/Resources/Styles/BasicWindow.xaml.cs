using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace CellUniverse.Views.Resources.Styles {

    //internal static class LocalExtensions {

    //    public static void ForTemplatedWindow(this object templateFrameworkElement, Action<System.Windows.Window> action) {
    //        System.Windows.Window window = ((FrameworkElement)templateFrameworkElement).TemplatedParent as System.Windows.Window;
    //        if (window != null) action(window);
    //    }
    //}

    public partial class BasicWindow {

        private const string WindowPaddingName = "PART_WindowPadding";
        private Window window = null;

        private object stateLocker = new object();
        private bool inChaging = false;

        //void IconMouseLeftButtonDown(object sender, MouseButtonEventArgs e) {
        //    if (e.ClickCount > 1)
        //        sender.ForTemplatedWindow(w => w.Close());
        //}

        //void CaptionMouseLeftButtonDown(object sender, MouseButtonEventArgs e) {
        //    if (e.ClickCount > 1)
        //        MaximizeButtonClick(sender, e);
        //}

        //void IconMouseUp(object sender, MouseButtonEventArgs e) {
        //    var element = sender as FrameworkElement;
        //    var point = element.PointToScreen(new Point(element.ActualWidth / 2, element.ActualHeight));
        //    sender.ForTemplatedWindow(w => SystemCommands.ShowSystemMenu(w, point));
        //}

        void WindowLoaded(object sender, RoutedEventArgs e) {
            window = (Window)sender;
            window.StateChanged += WindowStateChanged;
        }

        void WindowStateChanged(object sender, EventArgs e) {
            var containerBorder = (Border)window.Template.FindName(WindowPaddingName, window);

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
                        SystemParameters.PrimaryScreenWidth - SystemParameters.WorkArea.Right + 8,
                        SystemParameters.PrimaryScreenHeight - SystemParameters.WorkArea.Bottom + 7);
            }
        }

        //void MinimizeButtonClick(object sender, RoutedEventArgs e) {
        //    sender.ForTemplatedWindow(w => w.WindowState = WindowState.Minimized);
        //}

        //void MaximizeButtonClick(object sender, RoutedEventArgs e) {
        //    sender.ForTemplatedWindow(w => {
        //        if (w.WindowState == WindowState.Maximized) { w.WindowState = WindowState.Normal; }
        //        else { w.WindowState = WindowState.Maximized; }
        //        e.Handled = true;
        //    });
        //}

        //void CloseButtonClick(object sender, RoutedEventArgs e) {
        //    sender.ForTemplatedWindow(w => w.Close());
        //}
    }
}