/**
 * Kiki Agent Framework - 类名合并工具
 *
 * 类似于 clsx 或 classnames 的轻量实现
 */
export type ClassValue =
  | string
  | number
  | boolean
  | undefined
  | null
  | ClassArray
  | ClassDictionary;

interface ClassDictionary {
  [id: string]: any;
}

interface ClassArray extends Array<ClassValue> {}

function toVal(mix: ClassValue): string {
  let str = '';

  if (typeof mix === 'string' || typeof mix === 'number') {
    str += mix;
  } else if (typeof mix === 'object') {
    if (Array.isArray(mix)) {
      for (let k = 0; k < mix.length; k++) {
        if (mix[k]) {
          const y = toVal(mix[k]);
          if (y) {
            str && (str += ' ');
            str += y;
          }
        }
      }
    } else {
      for (const k in mix) {
        if (mix[k]) {
          str && (str += ' ');
          str += k;
        }
      }
    }
  }

  return str;
}

/**
 * 合并类名工具函数
 *
 * @example
 * classNames('foo', 'bar') // 'foo bar'
 * classNames('foo', { bar: true, baz: false }) // 'foo bar'
 * classNames(['foo', 'bar']) // 'foo bar'
 * classNames({ foo: true, bar: false }) // 'foo'
 */
export function classNames(...classes: ClassValue[]): string {
  let str = '';
  for (let i = 0; i < classes.length; i++) {
    const y = toVal(classes[i]);
    if (y) {
      str && (str += ' ');
      str += y;
    }
  }
  return str;
}

export default classNames;
