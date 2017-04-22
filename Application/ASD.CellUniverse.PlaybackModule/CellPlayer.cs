using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using Prism.Commands;

namespace ASD.CellUniverse.PlaybackModule {

    using FSM;

    public class CellPlayer : Control {

        private StateMachine stateMachine = new StateMachine();

        public Action Started {
            get => (Action)GetValue(StartedProperty);
            set => SetValue(StartedProperty, value);
        }

        public static readonly DependencyProperty StartedProperty = DependencyProperty.Register(
            "Started", typeof(Action), typeof(CellPlayer), new FrameworkPropertyMetadata(
                new Action(() => { }), OnStartedPropertyChanged));

        private static void OnStartedPropertyChanged(DependencyObject d, DependencyPropertyChangedEventArgs e) {

        }

        public CellPlayer() {
            stateMachine = new StateMachine();
            stateMachine.Started += OnStart;
            stateMachine.Paused += OnPause;
            stateMachine.Stopped += OnStop;
        }

        private void OnStart() { }
        private void OnPause() { }
        private void OnStop() { }

        static CellPlayer() {
            DefaultStyleKeyProperty
                .OverrideMetadata(typeof(CellPlayer), new FrameworkPropertyMetadata(typeof(CellPlayer)));
        }
    }
}