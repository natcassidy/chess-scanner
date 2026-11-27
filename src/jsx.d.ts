import 'preact';

declare module 'preact' {
  namespace JSX {
    interface IntrinsicElements {
      /** chessground piece sprite (styled by chessground.cburnett.css) */
      piece: HTMLAttributes<HTMLElement>;
    }
  }
}
